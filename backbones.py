import torch

import torch.nn as nn

from torchvision.ops.misc import FrozenBatchNorm2d

from torchvision.models import resnet, detection, segmentation

import timm


# https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.FrozenBatchNorm2d.convert_frozen_batchnorm
@torch.no_grad()
def convert_frozen_batchnorm(module):
    bn_module = (
        nn.modules.batchnorm.BatchNorm2d,
        nn.modules.batchnorm.SyncBatchNorm
    )
    res = module
    if isinstance(module, bn_module):
        res = FrozenBatchNorm2d(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = convert_frozen_batchnorm(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def get_backbone(backbone, pretrained=True):
    if backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        # pretrained on ImageNet for classification
        model = resnet.__dict__[backbone](
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )
    elif backbone == 'resnet50d':
        # pretrained on COCO for detection
        model = convert_frozen_batchnorm(
            detection.fasterrcnn_resnet50_fpn(pretrained=pretrained).backbone.body
        )
    elif backbone == 'resnet50s':
        # pretrained on COCO for segmentation
        model = convert_frozen_batchnorm(
            segmentation.deeplabv3_resnet50(pretrained=pretrained).backbone
        )
    elif backbone == 'resnet101s':
        # pretrained on COCO for segmentation
        model = convert_frozen_batchnorm(
            segmentation.deeplabv3_resnet101(pretrained=pretrained).backbone
        )

    elif backbone in ('cspdarknet53', 'efficientnet-b0', 'efficientnet-b3'):
        # model = convert_frozen_batchnorm(
        #     timm.create_model(
        #         backbone.replace('-', '_'),
        #         pretrained=True,
        #         features_only=True,
        #         #out_indices=(1, 2, 3, 4)
        #     )
        # )
        model = convert_frozen_batchnorm(
            timm.create_model(
                backbone.replace('-', '_'),
                pretrained=pretrained,
                num_classes=0,
                global_pool=''
            )
        )

    else:
        raise RuntimeError(f'{backbone} is not a valid backbone')

    # empty cache (dealloc modules other than the backbone)
    torch.cuda.empty_cache()

    return model
