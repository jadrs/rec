'''
detector-free referring expresion comprehension
'''
import os

import io

from parser import ArgumentParser

from PIL import Image

import torch

from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes, make_grid
from torchvision.transforms import ToTensor

from torch import nn

import pytorch_lightning as pl

import transformers

import matplotlib.pyplot as plt

from utils import cprint

from datasets import collate_fn, RefCLEF, RefCOCO, RefCOCOp, RefCOCOg, RegionDescriptionsVisualGnome

from transforms import get_transform, undo_box_transforms_batch, denormalize

import models as m

from encoders import get_tokenizer

from losses import GIoULoss, FocalLoss, SoftDiceLoss


class LitModel(pl.LightningModule):
    def __init__(self, model, beta, gamma, mu, learning_rate, weight_decay,
                 scheduler_param):
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.mu = mu
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.l1_loss = nn.SmoothL1Loss(reduction='mean', beta=beta)
        self.giou_loss = GIoULoss(reduction='mean')
        # self.segm_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.segm_loss = FocalLoss(reduction='mean')
        # self.segm_loss_2 = SoftDiceLoss(reduction='mean')

        self.scheduler_param = scheduler_param

        # self.save_hyperparameters()
        #     'beta', 'gamma', 'mu', 'learning_rate', 'weight_decay',
        #     'scheduler_param'
        # )

        # self.automatic_optimization = False

    @torch.no_grad()
    def peep(self, batch, preds, idxs=[0,]):
        N, _, H, W = batch['image'].size()
        size = torch.tensor([W, H, W, H], device=preds.device)

        imlist = []
        for i in idxs:
            image = (255 * denormalize(batch['image'])[i]).byte()
            boxes = torch.stack([batch['bbox'][i], preds[i]], dim=0) * size
            img = draw_bounding_boxes(image.cpu(), boxes.cpu(), colors=['blue', 'red'])

            plt.imshow(img.permute(1, 2, 0))
            plt.title(batch['expr'][i])
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg', bbox_inches='tight')
            buf.seek(0)

            img = ToTensor()(Image.open(buf))
            imlist.append(
                torch.nn.functional.interpolate(img.unsqueeze(0), (320, 320), mode='bilinear').squeeze(0)
            )

        return imlist

    @torch.no_grad()
    def iou(self, preds, targets):
        assert preds.size() == targets.size()
        preds = preds.unsqueeze(1)  # Nx1x4
        targets = targets.unsqueeze(1)  # Nx1x4
        return torch.FloatTensor([
            box_iou(preds[i], targets[i])
            for i in range(preds.size(0))
        ])

    def loss(self, dbox, dmask):
        l1_loss = self.l1_loss(dbox['preds'], dbox['targets'])

        giou_loss = 0.0
        if self.gamma > 0.0:
            giou_loss = self.giou_loss(dbox['preds'], dbox['targets'])

        segm_loss = 0.0
        if dmask['targets'] is not None and self.mu > 0.0:
            segm_loss = self.segm_loss(dmask['preds'], dmask['targets'])

        loss = l1_loss + self.gamma * giou_loss + self.mu * segm_loss

        return loss, (l1_loss, giou_loss, segm_loss)

    def training_step(self, batch, batch_idx):
        preds, segm_mask = self.model(batch)

        # AMP
        preds = preds.to(batch['bbox'].dtype)
        if segm_mask is not None:
            segm_mask = segm_mask.to(batch['mask_bbox'].dtype)

        loss, loss_terms = self.loss(
            dbox={'preds': preds, 'targets': batch['bbox']},
            dmask={'preds': segm_mask, 'targets': batch['mask_bbox']}
        )

        l1_loss, giou_loss, segm_loss = loss_terms

        self.log('loss/train_l1', l1_loss.detach(), on_step=True, on_epoch=False)

        self.log('loss/train_giou', giou_loss.detach(), on_step=True, on_epoch=False)

        if segm_mask is not None and self.mu > 0.0:
            self.log('loss/train_segm', segm_loss.detach(), on_step=True, on_epoch=False)

        self.log('loss/train', loss.detach(), on_step=True, on_epoch=True)

        iou = self.iou(preds, batch['bbox'])
        self.log('iou/train', iou.mean().detach(), on_step=False, on_epoch=True)

        hits = (iou > 0.5).float()
        self.log('acc/train', hits.mean().detach(), on_step=False, on_epoch=True)

        # # ---
        # # SAM
        # # $ git clone https://github.com/davda54/sam

        # optimizer = self.optimizers()

        # # first forward-backward pass
        # self.manual_backward(loss, optimizer)
        # optimizer.first_step(zero_grad=True)

        # loss_2, _ = self.loss(
        #     dbox={'preds': preds, 'targets': batch['bbox']},
        #     dmask={'preds': segm_mask, 'targets': batch['mask_bbox']},
        #     dcontr={'preds': preds, 'preds_adv': preds_adv, 'targets': batch['bbox']}
        # )

        # # second forward-backward pass
        # self.manual_backward(loss_2, optimizer)
        # optimizer.second_step(zero_grad=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, segm_mask = self.model(batch)

        # AMP
        preds = preds.to(batch['bbox'].dtype)
        if segm_mask is not None:
            segm_mask = segm_mask.to(batch['mask_bbox'].dtype)

        loss, _ = self.loss(
            dbox={'preds': preds, 'targets': batch['bbox']},
            dmask={'preds': segm_mask, 'targets': batch['mask_bbox']}
        )

        self.log('loss/val', loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 2:  # skip dryrun
            idxs = list(range(0, preds.size(0), max(1, preds.size(0)//16)))
            grid = make_grid(self.peep(batch, preds, idxs=idxs), nrow=len(idxs))
            self.logger.experiment.add_image(
                'validation', grid, global_step=self.current_epoch
            )
            self.logger.experiment.flush()

        # to original image coordinates
        preds = undo_box_transforms_batch(preds, batch['tr_param'])

        # clamp to original image size
        h0, w0 = batch['image_size'].unbind(1)
        image_size = torch.stack([w0, h0, w0, h0], dim=1)
        preds = torch.clamp(preds, torch.zeros_like(image_size), image_size-1)

        iou = self.iou(preds, batch['bbox_raw'])
        self.log('iou/val', iou.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.25).float()
        self.log('acc/val25', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.50).float()
        self.log('acc/val', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.75).float()
        self.log('acc/val75', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        # fig = gradient_flow(self.model)
        # self.logger.experiment.add_figure('gradient flow', fig, 0)

        return loss

    def test_step(self, batch, batch_idx):
        preds, _ = self.model(batch)

        # AMP
        preds = preds.to(batch['bbox'].dtype)

        # to original coordinates
        preds = undo_box_transforms_batch(preds, batch['tr_param'])

        # clamp to original image size
        h0, w0 = batch['image_size'].unbind(1)
        image_size = torch.stack([w0, h0, w0, h0], dim=1)
        preds = torch.clamp(preds, torch.zeros_like(image_size), image_size-1)

        iou = self.iou(preds, batch['bbox_raw'])
        self.log('iou/test', iou.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        hits = (iou > 0.5).float()
        self.log('acc/test', hits.mean().detach(), on_step=False, on_epoch=True, sync_dist=True)

        return

    def configure_optimizers(self):
        slow_ids = self.model.slow_param_ids()

        slow_params = [
            p for p in self.parameters()
            if id(p) in slow_ids and p.requires_grad
        ]

        fast_params = [
            p for p in self.parameters()
            if id(p) not in slow_ids and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {'params': slow_params, 'lr': 0.1*self.learning_rate},
                {'params': fast_params},
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # import sys
        # sys.path.append('sam')
        # from sam import SAM
        # optimizer = SAM(
        #     [
        #         {'params': slow_params, 'lr': 0.1*self.learning_rate},
        #         {'params': fast_params},
        #     ],
        #     torch.optim.Adam,
        #     lr=self.learning_rate,
        #     #momentum=0.9
        # )

        if self.scheduler_param in (None, {}):
            return optimizer

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.scheduler_param['milestones'],
                gamma=self.scheduler_param['gamma']
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer, ], [scheduler, ]


def run(args):

    pl.seed_everything(args.seed)

    num_workers = 0 if args.num_workers is None else args.num_workers

    transformers.logging.set_verbosity_error()

    # ------------------------------------------------------------------------

    tokenizer = get_tokenizer(args.cache)

    if args.dataset == 'vg':
        vg = RegionDescriptionsVisualGnome(
            data_root='./VisualGnome',
            transform=get_transform('train', input_size=args.input_size),  # also for validation
            tokenizer=tokenizer,
            max_length=args.max_length,
            with_mask_bbox=bool(args.mu > 0.0),
        )
        n_train = int(0.9 * len(vg))
        n_val = max(0, len(vg) - n_train)
        datasets = torch.utils.data.random_split(
            vg, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        datasets = {'train': datasets[0], 'val': datasets[1]}
        ds_splits = ('train', 'val')

    else:
        if args.dataset == 'refclef':
            ds_class, ds_splits = RefCLEF, ('train', 'val', 'test')
        elif args.dataset == 'refcoco':
            ds_class, ds_splits = RefCOCO, ('train', 'val', 'testA', 'testB')
        elif args.dataset == 'refcoco+':
            ds_class, ds_splits = RefCOCOp, ('train', 'val', 'testA', 'testB')
        elif args.dataset == 'refcocog':
            ds_class, ds_splits = RefCOCOg, ('train', 'val', 'test')
        else:
            raise RuntimeError('invalid dataset')

        if args.debug:
            ds_splits = ds_splits[:2]  # train, val only

        datasets = {
            split: ds_class(
                split,
                transform=get_transform(split, input_size=args.input_size),
                tokenizer=tokenizer,
                max_length=args.max_length,
                with_mask_bbox=bool(args.mu > 0.0)
            ) for split in ds_splits
        }

    # data loaders
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=bool(split == 'train') or bool(split == 'trainval'),
            num_workers=num_workers,
            pin_memory=bool(torch.cuda.is_available() and args.gpus is not None),
            collate_fn=collate_fn,
            drop_last=bool('test' not in split),
            persistent_workers=bool(num_workers > 0),
        ) for split in ds_splits
    }

    pdata = 0.1 if args.debug else 1.0

    model = m.IntuitionKillingMachine(
        backbone=args.backbone,
        pretrained=True,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_conv=args.num_conv,
        dropout_p=args.dropout_p,
        segmentation_head=bool(args.mu > 0.0),
        mask_pooling=args.mask_pooling
    )

    if args.pretrained_model is not None:
        cprint(
            f'loading pre-trained weights from {args.pretrained_model}',
            color='blue'
        )
        checkpoint = torch.load(
            args.pretrained_model, map_location=lambda storage, loc: storage
        )
        # strip 'model.' from pl checkpoint and remove ranking head params
        state_dict = {
            k[len('model.'):]: v
            for k, v in checkpoint['state_dict'].items()
            if 'ranking_head' not in k
        }
        missing, _ = model.load_state_dict(state_dict, strict=False)
        # ensure the only missing keys are those of the segmentation head
        assert [k for k in missing if 'segm' not in k] == []

    # learning rate scheduler
    scheduler_param = {}
    if args.scheduler:
        scheduler_param = {
            'milestones': [int(p * args.max_epochs) for p in (0.6, 0.9)],
            'gamma': 0.1
        }

    # model
    lit_model = LitModel(
        model=model,
        beta=args.beta,
        gamma=args.gamma,
        mu=args.mu,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_param=scheduler_param
    )

    if args.checkpoint is not None:
        # continue training and logging on the same dir
        # WARNING: make sure you use the same model/trainer arguments
        output_dir = os.path.dirname(args.checkpoint)
    else:
        # output dir from input arguments
        output_dir = ArgumentParser.args_to_path(args, (
            '--dataset',
            '--max-length',
            '--input-size',
            '--backbone',
            # '--language-model',
            # '--dropout-p',
            '--num-heads',
            '--num-layers',
            '--num-conv',
            '--beta',
            '--gamma',
            '--mu',
            '--mask-pooling',
            '--learning-rate',
            '--weight-decay',
            '--batch-size',
            '--grad-steps',
            '--max-epochs',
            '--scheduler',
            '--early-stopping',
            '--amp',
            '--debug',
        ), values_only=True)
    os.makedirs(output_dir, exist_ok=True)
    cprint(f'{output_dir}', color='blue')

    # log arguments for future reference
    with open(output_dir + '.log', 'w') as fh:
        fh.write(f'{vars(args)}')

    logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir,
        name='',
        version='',
        default_hp_metric=False
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename='best',
        monitor='acc/val',
        mode='max',
        save_last=args.save_last,
        verbose=False,
        every_n_epochs=1,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='acc/val',
        min_delta=0.0,
        patience=5,
        verbose=False,
        mode='max'
    )

    callbacks = [lr_monitor_callback, ]
    if not args.debug:
        callbacks.append(checkpoint_callback)
        if args.early_stopping:
            callbacks.append(early_stopping_callback)

    profiler = None
    if args.profile:
        profiler = pl.profiler.PyTorchProfiler(
            # filename=os.path.join(args.cache, 'trainval.prof'),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
        )

    gpus, strategy = None, None
    if args.gpus is not None:
        gpus = [int(i) for i in args.gpus.split(',')]

        if not args.force_ddp and len(gpus) > 1:
            try:
                import fairscale
            except ModuleNotFoundError:
                raise ModuleNotFoundError('you need fairscale to train with multiple GPUs')
            strategy = pl.plugins.DDPShardedPlugin()
        else:
            strategy = pl.plugins.DDPPlugin(find_unused_parameters=True)

    trainer = pl.Trainer(
        profiler=profiler,
        gpus=gpus,
        max_epochs=args.max_epochs,
        benchmark=True,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        strategy=strategy,
        limit_train_batches=pdata,
        limit_val_batches=pdata,
        # gradient_clip_val=1.0,
        # enable_pl_optimizer=False,
        accumulate_grad_batches=args.grad_steps,
        enable_checkpointing=bool(not args.debug),
        precision=16 if args.amp else 32,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val'],
        ckpt_path=args.checkpoint
    )

    if args.debug:
        return

    for split in [s for s in ds_splits if s not in ('train', 'val')]:
        print(f'evaluating \'{split}\' split ...')
        trainer.test(
            dataloaders=loaders[split],
            # ckpt_path='best',
            ckpt_path=checkpoint_callback.best_model_path
        )


if __name__ == '__main__':
    parser = ArgumentParser('Detector-free grounding')
    parser.add_model_args()
    parser.add_data_args()
    parser.add_loss_args()
    parser.add_pretrainer_args(learning_mode=False)
    parser.add_trainer_args()
    parser.add_runtime_args()
    args = parser.parse_args()
    cprint(f'{vars(args)}', color='red')

    run(args)
