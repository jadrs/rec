import os

from collections import OrderedDict

import torch

import transformers

import torch.nn.functional as F

from torch import nn

from torchvision.models import detection

from backbones import get_backbone

from embeddings import Box8PositionEmbedding2D

EPS = 1e-5

TRANSFORMER_MODEL = 'bert-base-uncased'
# TRANSFORMER_MODEL = 'distilroberta-base'


def get_tokenizer(cache=None):
    if cache is None:
        return transformers.BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    model_path = os.path.join(cache, TRANSFORMER_MODEL)
    os.makedirs(model_path, exist_ok=True)

    if os.path.exists(os.path.join(model_path, 'config.json')):
        return transformers.BertTokenizer.from_pretrained(model_path)

    tokenizer = transformers.BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    tokenizer.save_pretrained(model_path)

    return tokenizer


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


class ImageEncoder(nn.Module):
    def __init__(self, backbone='resnet50', out_channels=256, pretrained=True,
                 freeze_pretrained=False, with_pos=True):
        super().__init__()

        model = get_backbone(backbone, pretrained)

        if pretrained and freeze_pretrained:
            for p in model.parameters():
                p.requires_grad = False

        if 'resnet' in backbone:
            self.backbone = detection.backbone_utils.IntermediateLayerGetter(
                model, return_layers=OrderedDict({'layer4': 'output'})
            )
            channels = 512 if backbone in ('resnet18', 'resnet34') else 2048

        elif backbone in ('cspdarknet53', 'efficientnet-b0', 'efficientnet-b3'):
            output_layer_name = list(model.named_children())[-1][0]
            self.backbone = detection.backbone_utils.IntermediateLayerGetter(
                model, return_layers=OrderedDict({output_layer_name: 'output'})
            )
            channels = {
                'cspdarknet53': 1024,
                'efficientnet-b0': 1280,
                'efficientnet-b3': 1536
            }[backbone]

        else:
            raise RuntimeError('not a valid backbone')

        in_channels = channels+8 if with_pos else channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), 1, bias=False),
            nn.GroupNorm(1, out_channels, eps=EPS),
            # nn.ReLU(inplace=True),
        )
        self.proj.apply(weight_init)

        self.pos_emb = None
        if with_pos:
            self.pos_emb = Box8PositionEmbedding2D(with_projection=False)

        self.out_channels = out_channels

    def forward(self, img, mask=None):
        x = self.backbone(img)['output']
        if self.pos_emb is not None:
            x = torch.cat([x, self.pos_emb(x)], dim=1)
        x = self.proj(x)  # NxDxHxW

        x_mask = None
        if mask is not None:
            _, _, H, W = x.size()
            x_mask = F.interpolate(mask, (H, W), mode='bilinear')
            x_mask = (x_mask > 0.5).long()

        return x, x_mask


class FPNImageEncoder(nn.Module):
    def __init__(self,
                 backbone='resnet50', out_channels=256, pretrained=True,
                 freeze_pretrained=False, with_pos=True):
        super().__init__()

        model = get_backbone(backbone, pretrained)

        if pretrained and freeze_pretrained:
            for p in model.parameters():
                p.requires_grad = False

        if 'resnet' in backbone:
            if backbone in ('resnet18', 'resnet34'):
                in_channels_list = [64, 128, 256, 512]
            else:
                in_channels_list = [256, 512, 1024, 2048]
            return_layers = OrderedDict({
                'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'
            })

        # elif backbone == 'cspdarknet53':
        #     in_channels_list = [128, 256, 512, 1024]
        #     return_layers = OrderedDict({
        #         '1':'0', '2':'1', '3':'2', '4':'3'
        #     })

        else:
            raise RuntimeError('not a valid backbone')

        self.backbone = model

        self.fpn = detection.backbone_utils.BackboneWithFPN(
            backbone=self.backbone,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

        self.fpn.fpn.extra_blocks = None   # removes the 'pool' layer added by default

        self.out_channels = out_channels

        in_channels = int(out_channels + float(with_pos) * 8)

        self.proj = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), 1, bias=False),
                nn.GroupNorm(1, out_channels, eps=EPS),
                # nn.ReLU(inplace=True),
            ) for level in return_layers.values()
        })
        self.proj.apply(weight_init)

        self.pos_emb = None
        if with_pos:
            self.pos_emb = Box8PositionEmbedding2D(with_projection=False)

    def forward(self, x, mask=None):
        x = self.fpn(x)

        # smallest feature map (eg. 16x16 for an input of 512x512 pixels)
        _, _, H, W = list(x.values())[-1].size()

        x_out = None
        for level, fmap in x.items():
            # fmap = torch.relu(fmap)  # FPN blocks end in a conv2d, w/o activ.
            if self.pos_emb is not None:
                fmap = torch.cat([fmap, self.pos_emb(fmap)], dim=1)  # +Pos
            fmap = self.proj[level](fmap)   # Conv+BN+ReLU
            fmap = F.interpolate(fmap, (H, W), mode='nearest')  # to a smaller size
            if x_out is None:
                x_out = fmap
            else:
                x_out += fmap

        x_mask = None
        if mask is not None:
            x_mask = F.interpolate(mask, (H, W), mode='bilinear')
            x_mask = (x_mask > 0.5).long()

        return x_out, x_mask


class TransformerImageEncoder(nn.Module):
    def __init__(self,
                 backbone='resnet50', out_channels=256, pretrained=True,
                 freeze_pretrained=False, num_heads=8, num_layers=6,
                 dropout_p=0.1):
        super().__init__()

        model = get_backbone(backbone, pretrained)

        if pretrained and freeze_pretrained:
            for p in model.parameters():
                p.requires_grad = False

        if 'resnet' in backbone:
            self.backbone = detection.backbone_utils.IntermediateLayerGetter(
                model, return_layers=OrderedDict({'layer4': 'output'})
            )
            channels = 512 if backbone in ('resnet18', 'resnet34') else 2048

        elif backbone in ('cspdarknet53', 'efficientnet-b0', 'efficientnet-b3'):
            output_layer_name = list(model.named_children())[-1][0]
            self.backbone = detection.backbone_utils.IntermediateLayerGetter(
                model, return_layers=OrderedDict({output_layer_name: 'output'})
            )
            channels = {
                'cspdarknet53': 1024,
                'efficientnet-b0': 1280,
                'efficientnet-b3': 1536
            }[backbone]

        else:
            raise RuntimeError('not a valid backbone')

        self.proj = nn.Sequential(
            nn.Conv2d(channels, out_channels, (1, 1), 1, bias=False),
            nn.GroupNorm(1, out_channels, eps=EPS),
            # nn.ReLU(inplace=True),
        )
        self.proj.apply(weight_init)

        from transformers_pos import (
            TransformerEncoder,
            TransformerEncoderLayer,
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=out_channels,
                nhead=num_heads,
                dropout=dropout_p,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.pos_emb = Box8PositionEmbedding2D(embedding_dim=out_channels)

        self.out_channels = out_channels

    def flatten(self, x):
        N, _, H, W = x.size()
        x = x.to(memory_format=torch.channels_last)
        x = x.permute(0, 2, 3, 1).view(N, H*W, -1)  # NxHWxD
        return x

    def forward(self, img, mask=None):
        x = self.backbone(img)['output']
        x = self.proj(x)  # NxDxHxW

        N, _, H, W = x.size()

        pos = self.pos_emb(x)  # NxDxHxW
        pos = self.flatten(pos)  # NxRxD

        x = self.flatten(x)  # NxRxD

        # visibility mask
        x_mask = None
        if mask is not None:
            x_mask = F.interpolate(mask, (H, W), mode='bilinear')
            x_mask = (x_mask > 0.5).long()

        if mask is None:
            x = self.encoder(x, pos=pos)  # NxRxD
        else:
            mask = self.flatten(x_mask).squeeze(-1)
            x = self.encoder(x, src_key_padding_mask=(mask==0), pos=pos)  # NxRxD

        x = x.permute(0, 2, 1).view(N, -1, H, W)  # NxDxHxW

        return x, x_mask


class LanguageEncoder(nn.Module):
    def __init__(self, out_features=256, dropout_p=0.2,
                 freeze_pretrained=False, global_pooling=True):
        super().__init__()
        self.language_model = transformers.AutoModel.from_pretrained(
            TRANSFORMER_MODEL
        )

        if freeze_pretrained:
            for p in self.language_model.parameters():
                p.requires_grad = False

        self.out_features = out_features

        self.proj = nn.Sequential(
            nn.Linear(768, out_features),
            nn.LayerNorm(out_features, eps=1e-5),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
        )
        self.proj.apply(weight_init)

        self.global_pooling = bool(global_pooling)

    def forward(self, z):
        res = self.language_model(
            input_ids=z['input_ids'],
            position_ids=None,
            attention_mask=z['attention_mask']
        )

        if self.global_pooling:
            z, z_mask = self.proj(res.pooler_output), None
        else:
            z, z_mask = self.proj(res.last_hidden_state), z['attention_mask']

        return z, z_mask


class RNNLanguageEncoder(nn.Module):
    def __init__(self,
                 model_type='gru', hidden_size=1024, num_layers=2,
                 out_features=256, dropout_p=0.2, global_pooling=True):
        super().__init__()
        self.embeddings = transformers.AutoModel.from_pretrained(
            TRANSFORMER_MODEL
        ).embeddings.word_embeddings
        self.embeddings.weight.requires_grad = True

        # self.dropout_emb = nn.Dropout(0.5)
        self.dropout_emb = nn.Dropout(dropout_p)

        assert model_type in ('gru', 'lstm')
        self.rnn = (nn.GRU if model_type == 'gru' else nn.LSTM)(
            input_size=self.embeddings.weight.size(1),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Sequential(
            nn.Linear(2*hidden_size, out_features),
            nn.LayerNorm(out_features, eps=1e-5),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
        )
        self.proj.apply(weight_init)

        self.out_features = out_features

        self.global_pooling = bool(global_pooling)
        assert global_pooling  # only w/ global pooling

    def forward(self, z):
        z_mask = z['attention_mask']

        z = self.dropout_emb(self.embeddings(z['input_ids']))
        z, h_n = self.rnn(z, None)

        if isinstance(self.rnn, nn.LSTM):
            h_n = h_n[0]

        # hidden states as (num_layers, num_directions, batch, hidden_size)
        h_n = h_n.view(self.rnn.num_layers, 2, z.size(0), self.rnn.hidden_size)

        # last hidden states
        h_n = h_n[-1].permute(1, 0, 2).reshape(z.size(0), -1)
        h_n = self.proj(h_n)
        return h_n, z_mask


class SimpleEncoder(nn.Module):
    def __init__(self, out_features=256, dropout_p=0.1, global_pooling=True):
        super().__init__()
        self.embeddings = transformers.AutoModel.from_pretrained(
            TRANSFORMER_MODEL
        ).embeddings.word_embeddings
        self.embeddings.weight.requires_grad = True

        # self.dropout_emb = nn.Dropout(0.5)
        self.dropout_emb = nn.Dropout(dropout_p)

        self.proj = nn.Sequential(
            nn.Linear(768, out_features),
            nn.LayerNorm(out_features, eps=1e-5),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
        )
        self.proj.apply(weight_init)

        self.out_features = out_features

        self.global_pooling = bool(global_pooling)
        assert not self.global_pooling  # only w/o global pooling

    def forward(self, z):
        z_mask = z['attention_mask']
        z = self.embeddings(z['input_ids'])
        z = self.proj(self.dropout_emb(z))
        # z[:, 0] = torch.mean(z[:, 1:], 1)
        return z, z_mask
