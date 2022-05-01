import torch

import torch.nn.functional as F

from torch import nn

from torchvision.ops import box_convert

import embeddings as emb

import encoders as enc

from encoders import weight_init


def conv3x3(in_channels, out_channels, num_groups=0):
    return nn.Sequential(
        # Conv2d w/o bias since BatchNorm2d/GroupNorm already accounts for it (affine=True)
        nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(out_channels) if num_groups < 1 else nn.GroupNorm(num_groups, out_channels),
        nn.ReLU(inplace=True),
    )


class IntuitionKillingMachine(nn.Module):
    def __init__(self,
                 backbone='resnet50', pretrained=True, embedding_size=256,
                 num_heads=8, num_layers=6, num_conv=4, dropout_p=0.1,
                 segmentation_head=True, mask_pooling=True):
        super().__init__()

        if backbone.endswith('+tr'):
            self.vis_enc = enc.TransformerImageEncoder(
                backbone=backbone.rstrip('+tr'),
                out_channels=embedding_size,
                pretrained=pretrained,
            )

        elif backbone.endswith('+fpn'):
            self.vis_enc = enc.FPNImageEncoder(
                backbone=backbone.rstrip('+fpn'),
                out_channels=embedding_size,
                pretrained=pretrained,
                with_pos=False
            )
        else:
            self.vis_enc = enc.ImageEncoder(
                backbone=backbone,
                out_channels=embedding_size,
                pretrained=pretrained,
                with_pos=False
            )

        # freeze ResNet stem
        if 'resnet' in backbone:
            self.vis_enc.backbone.conv1.requires_grad = False
            self.vis_enc.backbone.conv1.eval()

        self.vis_pos_emb = emb.LearnedPositionEmbedding2D(
            embedding_dim=embedding_size
        )

        self.lan_enc = enc.LanguageEncoder(
            out_features=embedding_size,
            global_pooling=False,
            dropout_p=dropout_p
        )

        self.lan_pos_emb = emb.LearnedPositionEmbedding1D(
            embedding_dim=embedding_size
        )

        from transformers_pos import (
            XTransformerEncoder,
            TransformerEncoder,
            TransformerEncoderLayer,
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # ---
        # CONV PRE-HEAD (NECK?)

        if num_conv > 0:
            self.pre_head = nn.Sequential(*[
                conv3x3(embedding_size, embedding_size) for _ in range(num_conv)
            ])
            self.pre_head.apply(weight_init)
        else:
            self.pre_head = nn.Identity()

        # ---
        # OUTPUT HEADS

        # box prediction
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 4, bias=True),
            nn.Sigmoid()
        )
        self.head.apply(weight_init)

        # box segmentation mask
        self.segm_head = None
        if segmentation_head:
            self.segm_head = nn.Sequential(
                nn.Conv2d(embedding_size, 1, (3, 3), 1, 1, bias=True),
                #nn.Sigmoid()
            )
            self.segm_head.apply(weight_init)

        # ---

        self.mask_pooling = bool(mask_pooling)

        if self.mask_pooling and self.segm_head is None:
            raise RuntimeError('mask pooling w/o a segmentation head does not makes sense')

        self.embedding_size = embedding_size

    # def slow_param_ids(self, **kwargs):
    #     return []

    def slow_param_ids(self, slow_visual_backbone=True, slow_language_backbone=True):
        ids = []

        if slow_visual_backbone:
            ids += [id(p) for p in self.vis_enc.backbone.parameters()]
            if hasattr(self.vis_enc, 'encoder'):  # +tr
                ids += [id(p) for p in self.vis_enc.encoder.parameters()]

        if slow_language_backbone:
            if isinstance(self.lan_enc, enc.LanguageEncoder):
                ids += [id(p) for p in self.lan_enc.language_model.parameters()]
            else:
                ids += [id(p) for p in self.lan_enc.embeddings.parameters()]

        return ids

    def flatten(self, x):
        N, D, H, W = x.size()
        x = x.to(memory_format=torch.channels_last)
        x = x.permute(0, 2, 3, 1).view(N, H*W, D)
        return x  # NxHWxD

    def unflatten(self, x, size):
        N, R, D = x.size()
        H, W = size
        assert R == H*W, 'wrong tensor size'
        x = x.permute(0, 2, 1).to(memory_format=torch.contiguous_format)
        x = x.view(N, D, H, W)
        return x  # NxDxHxW

    def forward(self, input):
        img, mask, tok = input['image'], input['mask'], input['tok']

        # ---
        # VISUAL EMBEDDINGS

        x, x_mask = self.vis_enc(img, mask)   # NxDxHxW, NxHxW
        x_pos = self.vis_pos_emb(x, x_mask)

        N, D, H, W = x.size()  # save dims before flatten

        x = self.flatten(x)  # NxRxD
        x_mask = self.flatten(x_mask).squeeze(-1)  # NxR
        x_pos = self.flatten(x_pos)   # NxRxD

        # ---
        # LANGUAGE EMBEDDINGS

        z, z_mask = self.lan_enc(tok)   # NxTxD, NxT
        z_pos = self.lan_pos_emb(z)  # NxTxD

        # ---
        # V+L TRANSFORMER

        # [...visual...]+[[CLS]...language tokens...[SEP]]
        xz = torch.cat([x, z], dim=1)
        xz_mask = torch.cat([x_mask, z_mask], dim=1)
        xz_pos = torch.cat([x_pos, z_pos], dim=1)

        xz = self.encoder(xz, src_key_padding_mask=(xz_mask==0), pos=xz_pos)  #, size=(H,W))

        # restore spatiality of visual embeddings after cross-modal encoding
        xz_vis = xz[:, :H*W, ...]
        xz_vis = self.unflatten(xz_vis, (H, W))

        x_mask = self.unflatten(x_mask.unsqueeze(-1), (H, W))

        # ---

        # convolutional pre-head
        xz_vis = self.pre_head(xz_vis)

        # ---

        # segmentation head w/ (opt.) pooling
        segm_mask, pooled_feat = None, None
        if self.segm_head is not None:
            segm_mask = torch.sigmoid(self.segm_head(xz_vis)) * x_mask
            if self.mask_pooling:  # box mask guided pooling
                pooled_feat = (segm_mask * xz_vis).sum((2, 3)) / segm_mask.sum((2, 3))
            segm_mask = F.interpolate(segm_mask, img.size()[2:], mode='bilinear', align_corners=True)

        # if not mask_pooling, do the pooling using all visual feats (equiv. to a uniform mask)
        if pooled_feat is None:
            pooled_feat = (x_mask * xz_vis).sum((2, 3)) / x_mask.sum((2, 3))

        # bbox prediction
        pred = self.head(pooled_feat)
        pred = box_convert(pred, 'cxcywh', 'xyxy')

        return pred, segm_mask
