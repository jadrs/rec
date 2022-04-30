import math

import torch

from torch import nn


# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionEmbedding1D(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=128):
        super().__init__()

        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # # x: Tensor, shape [batch_size, seq_len, embedding_dim]
        # x = x + self.pe[:, :x.size(1)]
        # return self.dropout(x)
        N, T, _ = x.size()
        return self.pe[:, :T].repeat(N, 1, 1)


class LearnedPositionEmbedding1D(nn.Module):
    def __init__(self, embedding_dim, max_len=128):
        super().__init__()
        self.pe = nn.Parameter(torch.Tensor(1, max_len, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.pe)

    def forward(self, x):
        N, T, _ = x.size()
        return self.pe[:, :T].repeat(N, 1, 1)


# https://huggingface.co/transformers/_modules/transformers/models/detr/modeling_detr.html
class PositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim, temperature=10000, normalize=False,
                 scale=None):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.half_embedding_dim = embedding_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        assert pixel_mask is not None, "No pixel mask provided"
        if pixel_mask.dim() == 4 and pixel_mask.size(1) == 1:
            pixel_mask = pixel_mask.squeeze(1)
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.half_embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.divide(dim_t, 2, rounding_mode='floor') / self.half_embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((
            pos_x[:, :, :, 0::2].sin(),
            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((
            pos_y[:, :, :, 0::2].sin(),
            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# https://huggingface.co/transformers/_modules/transformers/models/detr/modeling_detr.html
class LearnedPositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        assert embedding_dim % 2 == 0, 'embedding dimensionality must be even'
        self.rows_embeddings = nn.Embedding(50, embedding_dim//2)
        self.cols_embeddings = nn.Embedding(50, embedding_dim//2)

    def forward(self, pixel_values, pixel_mask=None):
        h, w = pixel_values.shape[-2:]
        i = torch.arange(w, device=pixel_values.device)
        j = torch.arange(h, device=pixel_values.device)
        x_emb = self.cols_embeddings(i)
        y_emb = self.rows_embeddings(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


class Box8PositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim, with_projection=True):
        super().__init__()

        self.proj = None
        if with_projection:
            self.proj = nn.Linear(8, embedding_dim)
            nn.init.xavier_normal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, fmap, fmap_mask=None):
        N, _, H, W = fmap.size()

        y1, x1 = torch.meshgrid(
            torch.arange(H, device=fmap.device, dtype=torch.float)/H,
            torch.arange(W, device=fmap.device, dtype=torch.float)/W
        )
        y2, x2 = x1+1.0/W, y1+1.0/H
        ww, hh = x2-x1, y2-y1
        # x1, y1 = 2*x1-1, 2*y1-1
        # x2, y2 = 2*x2-1, 2*y2-1
        xc, yc = x1+0.5/W, y1+0.5/H

        pos = torch.stack([x1, y1, x2, y2, xc, yc, ww, hh], dim=-1)
        if self.proj is not None:
            pos = self.proj(pos)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0).repeat(N, 1, 1, 1)
        return pos

    def encode_boxes(self, boxes):
        x1, y1, x2, y2 = boxes.unbind(-1)
        ww, hh = x2-x1, y2-y1
        xc, yc = x1+0.5*ww, y1+0.5*hh
        pos = torch.stack([x1, y1, x2, y2, xc, yc, ww, hh], dim=-1)
        if self.proj is not None:
            pos = self.proj(pos)
        return pos


class RelativePositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim, spatial_bins=(16, 16), with_projection=True):
        super().__init__()

        assert isinstance(spatial_bins, (list, tuple)) and len(spatial_bins) == 2
        self.spatial_bins = spatial_bins

        self.proj = None
        if with_projection:
            self.proj = nn.Linear(2*spatial_bins[0]*spatial_bins[1], embedding_dim)
            nn.init.xavier_normal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, fmap, fmap_mask=None):
        N, _, H, W = fmap.size()

        BH, BW = self.spatial_bins
        yc, xc = torch.meshgrid(
            0.5/BH + torch.arange(BH, device=fmap.device, dtype=torch.float)/BH,
            0.5/BW + torch.arange(BW, device=fmap.device, dtype=torch.float)/BW
        )

        pos = torch.stack([xc, yc], dim=-1).view(-1, 1, 2)
        pos = (pos - pos.transpose(0, 1)).reshape(BH, BW, -1)  # relative positions

        if self.proj is not None:
            pos = self.proj(pos)

        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)

        if H != BH or W != BW:
            pos = nn.functional.interpolate(pos, (H, W), mode='nearest')

        pos = pos.repeat(N, 1, 1, 1)

        return pos
