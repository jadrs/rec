import os

import argparse

from utils import timestamp


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, description):
        super(ArgumentParser, self).__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
            add_help=True,
            allow_abbrev=False
        )

    def add_model_args(self):
        group = self.add_argument_group('model')
        group.add_argument(
            '--backbone',
            help='Visual backbone. "+tr" adds an transformer after the CNN',
            type=str,
            default='resnet50',
            choices=(
                'resnet18', 'resnet34', 'resnet50', 'resnet101',  # imagenet
                'resnet18+tr', 'resnet34+tr', 'resnet50+tr', 'resnet101+tr',
                'resnet18+fpn', 'resnet34+fpn', 'resnet50+fpn', 'resnet101+fpn',
                'resnet50d',  # COCO detection
                'resnet50d+tr',
                'resnet50d+fpn',
                'resnet50s', 'resnet101s',  # COCO segmentation
                'resnet50s+tr', 'resnet101s+tr',
                'resnet50s+fpn', 'resnet101s+fpn',
                'cspdarknet53',  # timm
                'efficientnet-b0', 'efficientnet-b3',
            ),
        )
        group.add_argument(
            '--mask-pooling',
            help='if set, pool visual features using a mask',
            action='store_true'
        )
        group.add_argument(
            '--dropout-p',
            help='Dropout p',
            type=float,
            default=0.1,
        )
        group.add_argument(
            '--num-heads',
            help='number of heads for the cross-modal encoder',
            type=int,
            default=8,
        )
        group.add_argument(
            '--num-layers',
            help='number of layers for the cross-modal encoder',
            type=int,
            default=6,
        )
        group.add_argument(
            '--num-conv',
            help='number of convolutional blocks (post transformer)',
            type=int,
            default=8,
        )

    def add_data_args(self):
        group = self.add_argument_group('data')
        group.add_argument(
            '--dataset',
            help='dataset',
            type=str,
            default='refcoco',
            choices=('refclef', 'refcoco', 'refcoco+', 'refcocog', 'vg')
        )
        group.add_argument(
            '--max-length',
            help='max token sequence length',
            type=int,
            default=32
        )
        group.add_argument(
            '--input-size',
            help='images will be resized to INPUT_SIZExINPUT_SIZE pixels',
            type=int,
            default=512
        )

    def add_loss_args(self):
        group = self.add_argument_group('loss function')
        group.add_argument(
            '--beta',
            help='smooth L1 loss beta parameter',
            default=0.1,
            type=float
        )
        group.add_argument(
            '--gamma',
            help='GIoU loss term weight',
            default=0.1,
            type=float
        )
        group.add_argument(
            '--mu',
            help='box segmentation term weight',
            default=0.1,
            type=float
        )

    def add_trainer_args(self):
        group = self.add_argument_group('trainer')
        group.add_argument(
            '--learning-rate',
            help='learning rate',
            default=1e-4,
            type=float
        )
        group.add_argument(
            '--weight-decay',
            help='weight decay',
            default=0.0,
            type=float
        )
        group.add_argument(
            '--batch-size',
            help='batch size',
            default=16,
            type=int
        )
        group.add_argument(
            '--grad-steps',
            help='accumulates gradient every GRAD_STEPS batches',
            default=1,
            type=int
        )
        group.add_argument(
            '--max-epochs',
            help='max number of epochs',
            default=50,
            type=int
        )
        group.add_argument(
            '--scheduler',
            help='use a multistep scheduler (no warmup)',
            action='store_true'
        )

    def add_runtime_args(self):
        group = self.add_argument_group('runtime arguments')
        group.add_argument(
            '--gpus',
            help='GPUs identifiers',
            type=str
        )
        # group.add_argument(
        #     '--num-threads',
        #     help='torch num threads',
        #     type=int
        # )
        group.add_argument(
            '--num-workers',
            help='dataloader num workers',
            type=int
        )
        group.add_argument(
            '--seed',
            help='random seed',
            type=int,
            default=3407  # https://arxiv.org/pdf/2109.08203v1.pdf :-)
        )
        group.add_argument(
            '--suffix',
            help='path suffix',
            type=str
        )
        group.add_argument(
            '--cache',
            help='cache path',
            type=str,
            default='./cache'
        )
        group.add_argument(
            '--debug',
            help='if set, run on a small percentage of the (training) data',
            action='store_true',
        )
        group.add_argument(
            '--early-stopping',
            help='if set, enables the early stopping callback',
            action='store_true',
        )
        group.add_argument(
            '--amp',
            help='if set, enables automatic mixed precision (AMP) training',
            action='store_true',
        )
        group.add_argument(
            '--force-ddp',
            help='if set, force strategy=DDP',
            action='store_true',
        )
        group.add_argument(
            '--profile',
            help='if set, enables profiling',
            action='store_true',
        )
        group.add_argument(
            '--checkpoint',
            help='resume training from CHECKPOINT',
            type=str
        )
        group.add_argument(
            '--save-last',
            help='if set, allways save last epoch checkpoint',
            action='store_true',
        )

    @staticmethod
    def args_to_path(args, keys, values_only=False):
        path = os.path.join(os.path.abspath(args.cache), timestamp())

        keys = [k.lstrip('-').replace('-', '_') for k in keys if k not in ('', None)]

        vargs = vars(args)

        for k in keys:
            if k == 'suffix':
                continue
            if values_only:
                if type(vargs[k]) is bool:
                    path += f'_{int(vargs[k])}'  # _0 or _1 for a bool var
                else:
                    path += f'_{vargs[k]}'
            else:
                if type(vargs[k]) is bool and vargs[k]:  # if bool and set
                    path += f'_{k.replace("_", "-")}'
                else:
                    path += f'_{k.replace("_", "-")}_{vargs[k]}'

        if args.suffix is not None:
            path += f'_{args.suffix}'

        return path
