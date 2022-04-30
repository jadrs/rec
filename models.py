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

        # self.vis_pos_emb = emb.PositionEmbedding2D(
        #     embedding_dim=embedding_size
        # )

        # self.vis_pos_emb = emb.Box8PositionEmbedding2D(
        #     embedding_dim=embedding_size
        # )

        # self.vis_pos_emb = emb.RelativePositionEmbedding2D(
        #     embedding_dim=embedding_size
        # )

        # self.vis_pos_scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0))

        self.lan_enc = enc.LanguageEncoder(
            out_features=embedding_size,
            global_pooling=False,
            dropout_p=dropout_p
        )

        # self.lan_enc = enc.SimpleEncoder(
        #     out_features=embedding_size,
        #     global_pooling=False,
        #     dropout_p=dropout_p
        # )

        self.lan_pos_emb = emb.LearnedPositionEmbedding1D(
            embedding_dim=embedding_size
        )

        # self.lan_pos_emb = emb.PositionEmbedding1D(
        #     embedding_dim=embedding_size
        # )

        # self.lan_pos_scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0))

        # self.reg_emb = nn.Embedding(1, embedding_size, dtype=torch.float)
        # self.reg_emb.apply(weight_init)

        # self.mod_emb = nn.Embedding(2, embedding_size, dtype=torch.float)

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

        # self.encoder = XTransformerEncoder(
        #     TransformerEncoderLayer(
        #         d_model=embedding_size,
        #         nhead=num_heads,
        #         dropout=dropout_p,
        #         batch_first=True
        #     ),
        #     num_layers=num_layers,
        #     num_conv=2
        # )

        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=embedding_size,
        #         nhead=num_heads,
        #         dropout=dropout_p,
        #         batch_first=True
        #     ),
        #     num_layers=num_layers
        # )

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

        # # add modality embeddings to the pos embeddings so that it gets added
        # # on each layer of the transformers
        # x_pos = x_pos + self.mod_emb.weight[0].view(1, 1, -1)
        # z_pos = z_pos + self.mod_emb.weight[1].view(1, 1, -1)

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

        # # segm head before conv head. Just to constraint the cross-modal encoder
        # segm_mask, pooled_feat = None, None
        # if self.segm_head is not None:
        #     segm_mask = self.segm_head(xz_vis) * x_mask
        #     assert not self.mask_pooling
        #     segm_mask = F.interpolate(segm_mask, img.size()[2:], mode='bilinear', align_corners=True)

        # convolutional pre-head
        xz_vis = self.pre_head(xz_vis)

        # # cat average language embedding to visual fmap
        # xz_lan = xz[:, H*W:, ...].mean(1).view(N, -1, 1, 1).repeat(1, 1, H, W)
        # xz_vis = torch.cat([xz_vis, xz_lan], dim=1)

        # ---

        # segmentation head w/ (opt.) pooling
        segm_mask, pooled_feat = None, None
        if self.segm_head is not None:
            #segm_mask = self.segm_head(xz_vis) * x_mask
            segm_mask = torch.sigmoid(self.segm_head(xz_vis)) * x_mask
            if self.mask_pooling:  # box mask guided pooling
                pooled_feat = (segm_mask * xz_vis).sum((2, 3)) / segm_mask.sum((2, 3))
            segm_mask = F.interpolate(segm_mask, img.size()[2:], mode='bilinear', align_corners=True)

        # if not mask_pooling, do the pooling using all visual feats (equiv. to a uniform mask)
        if pooled_feat is None:
            # pooled_feat = xz_vis.mean((2, 3))
            pooled_feat = (x_mask * xz_vis).sum((2, 3)) / x_mask.sum((2, 3))

        # bbox prediction
        pred = self.head(pooled_feat)
        pred = box_convert(pred, 'cxcywh', 'xyxy')

        return pred, segm_mask


class IntuitionKillingMachine_pretraining(IntuitionKillingMachine):
    def __init__(self, backbone='resnet50', pretrained=True, embedding_size=256,
                 num_heads=8, num_layers=6, dropout_p=0.1, num_classes=0):
        super().__init__(
            backbone=backbone, pretrained=pretrained, embedding_size=embedding_size,
            num_heads=num_heads, num_layers=num_layers, dropout_p=dropout_p,
            segmentation_head=False, mask_pooling=False
        )

        # self.vis_enc = ImageEncoder(
        #     backbone=backbone,
        #     out_channels=embedding_size,
        #     pretrained=False,
        #     with_pos=False
        # )

        # self.vis_enc.backbone.conv1.requires_grad = False
        # self.vis_enc.backbone.conv1.eval()

        # self.lan_enc.language_model.requires_grad = False
        # self.lan_enc.language_model.eval()

        # ---
        # AUXILIARY TASK HEADS

        self.ranking_head = nn.Sequential(
            nn.Linear(embedding_size, 1, bias=True),
        )
        self.ranking_head.apply(weight_init)

        self.cls_head = None
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(embedding_size, num_classes, bias=True),
            )
            self.cls_head.apply(weight_init)

    def embedd(self, img, mask, tok):
        x, x_mask = self.vis_enc(img, mask)   # NxDxHxW, Nx1xHxW
        x_pos = self.vis_pos_emb(x, x_mask)

        N, D, H, W = x.size()

        x = self.flatten(x)  # NxRxD
        x_mask = self.flatten(x_mask).squeeze(-1)  # NxR
        x_pos = self.flatten(x_pos)

        K, _, T = tok['input_ids'].size()  # already transposed

        output = {
            # 'x_pre_avg': (x_mask.unsqueeze(-1) * x).sum(1) / x_mask.unsqueeze(-1).sum(1),
            'x_pre': x,
            'z_pre': [],
            'x_post': [],
            'z_post': [],
        }

        for k in range(K):
            z, z_mask = self.lan_enc({
                'input_ids': tok['input_ids'][k],
                'attention_mask': tok['attention_mask'][k],
            })  # NxTxD, NxT
            z_pos = self.lan_pos_emb(z)

            # # [CLS] language embedding
            # z_pre = z[:, 0]

            # avg. embedding
            z_pre = (z * z_mask.unsqueeze(-1)).sum(1) / z_mask.unsqueeze(-1).sum(1)

            # # avg. embedding w/o [CLS] [SEP]
            # z_mask_ = torch.roll(z_mask, -1, 1)  # zero last token ([SEP])
            # z_mask_[:, [0, -1]] = 0  # zero first token ([CLS]) and ensure the last one from a full attended sequence is also zeroed
            # z_pre = (z * z_mask_.unsqueeze(-1)).sum(1) / z_mask_.unsqueeze(-1).sum(1)

            output['z_pre'].append(z_pre)

            # transformer
            xz = torch.cat([x, z], dim=1)
            xz_mask = torch.cat([x_mask, z_mask], dim=1)
            xz_pos = torch.cat([x_pos, z_pos], dim=1)

            xz = self.encoder(xz, src_key_padding_mask=(xz_mask==0), pos=xz_pos)

            # convolutional pre-head
            xz_vis = xz[:, :H*W, ...]
            xz_vis = self.unflatten(xz_vis, (H, W))
            xz_vis = self.pre_head(xz_vis)
            xz_vis = self.flatten(xz_vis)

            x_post = (x_mask.unsqueeze(-1) * xz_vis).sum(1) / x_mask.unsqueeze(-1).sum(1)
            output['x_post'].append(x_post)

            # w/ [CLS] and [SEP]
            z_post = (z_mask.unsqueeze(-1) * xz[:, H*W:, ...]).sum(1) / z_mask.unsqueeze(-1).sum(1)
            output['z_post'].append(z_post)

        output['z_pre'] = torch.stack(output['z_pre'], dim=1)  # NxKxD
        output['x_post'] = torch.stack(output['x_post'], dim=1)  # NxKxD
        output['z_post'] = torch.stack(output['z_post'], dim=1)  # NxKxD

        return output

    # def embedd(self, img, mask, tok):
    #     x, x_mask = self.vis_enc(img, mask)   # NxDxHxW, NxHxW
    #     x_pos = self.vis_pos_emb(x, x_mask.squeeze(1))

    #     N, D, H, W = x.size()
    #     v_size = (N, D, H, W)

    #     x = self.flatten(x)  # NxRxD
    #     x_mask = self.flatten(x_mask).squeeze(-1)  # NxR
    #     x_pos = self.flatten(x_pos)

    #     K, _, T = tok['input_ids'].size()  # already transposed

    #     embs = []
    #     for k in range(K):
    #         z, z_mask = self.lan_enc({
    #             'input_ids': tok['input_ids'][k],
    #             'attention_mask': tok['attention_mask'][k],
    #         })  # NxTxD, NxT
    #         z_pos = self.lan_pos_emb(z)

    #         embs.append(z[:, 0, ...])  # [CLS] -> noun chunk embedding

    #     embs = torch.stack(embs, dim=1)  # NxKxD

    #     # [...visual...]+[[nc1]...[ncK]]
    #     embs = torch.cat([x, embs], dim=1)
    #     mask = torch.cat([x_mask, torch.ones(N, K, dtype=torch.int64, device=x.device)], dim=1)
    #     pos = torch.cat([x_pos, torch.zeros((N, K, D), dtype=torch.float32, device=x.device)], dim=1)

    #     embs = self.encoder(embs, src_key_padding_mask=(mask==0), pos=pos)  # Nx(HW+K)xD

    #     embs = F.normalize(embs, p=2, dim=-1)

    #     return embs, v_size  # Nx(HW+K)xD

    def forward(self, input):
        img, mask, tok = input['image'], input['mask'], input['tok']

        tok = {
            'input_ids': tok['input_ids'].transpose(0, 1),  # KxNxT
            'attention_mask': tok['attention_mask'].transpose(0, 1),  # KxNxT
        }

        res = self.embedd(img, mask, tok)

        logits = None
        if self.cls_head is not None:
            logits = self.cls_head(res['x_pre'].mean(1))

        # xz_rnk = torch.cat([
        #     res['x_post'], res['z_post']
        # ], dim=-1)
        #
        # xz_rnk = torch.cat([
        #     F.normalize(res['x_post'], p=2, dim=1),
        #     F.normalize(res['z_post'], p=2, dim=1)
        # ], dim=-1)
        #
        # xz_rnk = res['x_post'] + res['z_post']
        #
        xz_rnk = res['x_post']

        xz_rnk = F.normalize(xz_rnk, p=2, dim=1)
        scores = self.ranking_head(xz_rnk).squeeze(-1)  # NxK

        return scores, logits, (res['x_pre'], res['z_pre'])


if __name__ == '__main__':
    x = torch.rand((2, 3, 512, 512))
    net = enc.ImageEncoder(backbone='resnet18')
    x, x_mask = net(x)
    print(x.size(), x_mask.size(), x_mask.dtype)

    net = enc.LanguageEncoder(32)
    z, z_mask = net(['hello world', 'hola mundo'])
    print(z.size(), z_mask.size(), z_mask.dtype)
