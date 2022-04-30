import torch
from torch import nn
from torch.nn import functional as F


# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
def giou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630
    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class GIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return giou_loss(pred, target, self.reduction, eps=1e-7)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        # self.ce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        logpt = -self.ce_loss(pred, target)   # if y=1: pt=p, else: pt=1-p
        pt = torch.exp(logpt)
        loss = ((1.0 - pt) ** self.gamma) * (-logpt)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # working with logits
        loss = 1.0 - 2 * pred * target / (pred ** 2 + target ** 2 + 2**-23)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ListNetLoss(nn.Module):
    # Cao et al. (2007) Learning to Rank: From Pairwise Approach to Listwise Approach
    def __init__(self, tau=1.0, reduction='mean'):
        super().__init__()
        self.tau = tau  # target temperature
        self.reduction = reduction

    def forward(self, pred, target):
        p1 = F.softmax(self.tau * target, dim=1)
        log_p2 = F.log_softmax(pred, dim=1)
        loss = -(p1 * log_p2).sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GroundingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    # Zareian et al (2021) Open-Vocabulary Object Detection Using Captions
    def score(self, x, z):
        sim = torch.einsum('nik,njk->nij', x, z)  # NxRxK
        sim = torch.softmax(sim, dim=1) * sim
        sim = sim.sum((1,2)) / z.size(1)  # N
        return sim

    def forward(self, embs):
        x, z = embs  # Nx*xD

        sim_x = torch.stack([
            self.score(torch.roll(x, i, 0), z)
            for i in range(x.size(0))
        ], dim=0)  # NxN
        loss_x = -torch.log_softmax(sim_x, dim=0)[0]  # N

        sim_z = torch.stack([
            self.score(x, torch.roll(z, j, 0))
            for j in range(z.size(0))
        ], dim=0)  # NxN
        loss_z = -torch.log_softmax(sim_z, dim=0)[0]  # N

        loss = loss_x + loss_z
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BoxContrastiveLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=0.0):
        super().__init__()
        self.reduction = reduction
        self.gamma = float(gamma)

    def forward(self, pred, pred_alt, target):

        # l1_loss = F.smooth_l1_loss(pred, target, reduction='none')
        # l1_loss_alt = F.smooth_l1_loss(pred_alt, target, reduction='none')

        # iou_loss, iou_loss_alt = 0.0, 0.0
        # if self.gamma > 0.0:
        #     iou_loss = giou_loss(pred, target, reduction='none')
        #     iou_loss_alt = giou_loss(pred_alt, target, reduction='none')

        # ell = torch.exp(-(l1_loss + self.gamma * iou_loss).squeeze())
        # ell_alt = torch.exp(-(l1_loss_alt + self.gamma * iou_loss_alt).squeeze())

        ell = 1.0 - giou_loss(pred, target, 'none', eps=1e-7).squeeze()
        ell_alt = 1.0 - giou_loss(pred_alt, target, 'none', eps=1e-7).squeeze()

        loss = -torch.log_softmax(torch.stack([ell, ell_alt], dim=0), dim=0)[0]

        # loss = torch.relu(self.gamma - ell + ell_alt)   # constraint: ell - ell_alt > gamma

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
