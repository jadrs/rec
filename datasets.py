import os

import json

import random

import torch

import ijson

import numpy as np

from PIL import Image

from torchvision.transforms import ToTensor

from torchvision.ops import box_convert, clip_boxes_to_image

from re_classifier import REClassifier

from utils import progressbar


def collate_fn(batch):
    image = torch.stack([s['image'] for s in batch], dim=0)

    image_size = torch.FloatTensor([s['image_size'] for s in batch])

    # bbox = torch.stack([s['bbox'] for s in batch], dim=0)
    bbox = torch.cat([s['bbox'] for s in batch], dim=0)

    # bbox_raw = torch.stack([s['bbox_raw'] for s in batch], dim=0)
    bbox_raw = torch.cat([s['bbox_raw'] for s in batch], dim=0)

    expr = [s['expr'] for s in batch]

    tok = None
    if batch[0]['tok'] is not None:
        tok = {
            'input_ids': torch.cat([s['tok']['input_ids'] for s in batch], dim=0),
            'attention_mask': torch.cat([s['tok']['attention_mask'] for s in batch], dim=0)
        }

        # dynamic batching
        max_length = max([s['tok']['length'] for s in batch])
        tok = {
            'input_ids': tok['input_ids'][:, :max_length],
            'attention_mask': tok['attention_mask'][:, :max_length],
        }

    mask = None
    if batch[0]['mask'] is not None:
        mask = torch.stack([s['mask'] for s in batch], dim=0)

    mask_bbox = None
    if batch[0]['mask_bbox'] is not None:
        mask_bbox = torch.stack([s['mask_bbox'] for s in batch], dim=0)

    tr_param = [s['tr_param'] for s in batch]

    return {
        'image': image,
        'image_size': image_size,
        'bbox': bbox,
        'bbox_raw': bbox_raw,
        'expr': expr,
        'tok': tok,
        'tr_param': tr_param,
        'mask': mask,
        'mask_bbox': mask_bbox,
    }


class RECDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, tokenizer=None, max_length=32, with_mask_bbox=False):
        super().__init__()
        self.samples = []  # list of samples: [(file_name, expresion, bbox)]
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.with_mask_bbox = bool(with_mask_bbox)

    def tokenize(self, inp, max_length):
        return self.tokenizer(
            inp,
            return_tensors='pt',
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length
        )

    def print_stats(self):
        print(f'{len(self.samples)} samples')
        lens = [len(expr.split()) for _, expr, _ in self.samples]
        print('expression lengths stats: '
              f'min={np.min(lens):.1f}, '
              f'mean={np.mean(lens):.1f}, '
              f'median={np.median(lens):.1f}, '
              f'max={np.max(lens):.1f}, '
              f'99.9P={np.percentile(lens, 99.9):.1f}'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, expr, bbox = self.samples[idx]

        if not os.path.exists(file_name):
            raise IOError(f'{file_name} not found')
        img = Image.open(file_name).convert('RGB')

        # if isinstance(expr, (list, tuple)):
        #     expr = random.choice(expr)

        # image size as read from disk (PIL)
        W0, H0 = img.size

        # # ensure box coordinates fall inside the image
        # bbox = clip_boxes_to_image(bbox, (H0, W0))
        # assert torch.all(bbox[:, (0, 1)] <= bbox[:, (2, 3)])  # xyxy format

        sample = {
            'image': img,
            'image_size': (H0, W0),  # image original size
            'bbox': bbox.clone(),  # box transformations are inplace ops
            'bbox_raw': bbox.clone(),  # raw boxes w/o any transformation (in pixels)
            'expr': expr,
            'tok': None,
            'mask': torch.ones((1, H0, W0), dtype=torch.float32),  # visibiity mask
            'mask_bbox': None,  # target bbox mask
        }

        # apply transforms
        if self.transform is None:
            sample['image'] = ToTensor()(sample['image'])
        else:
            sample = self.transform(sample)

        # tokenize after the transformations (just in case there where a left<>right substitution)
        if self.tokenizer is not None:
            sample['tok'] = self.tokenize(sample['expr'], self.max_length)
            sample['tok']['length'] = sample['tok']['attention_mask'].sum(1).item()

        # bbox segmentation mask
        if self.with_mask_bbox:
            # image size after transforms
            _, H, W = sample['image'].size()

            # transformed bbox in pixels
            bbox = sample['bbox'].clone()
            bbox[:, (0, 2)] *= W
            bbox[:, (1, 3)] *= H
            bbox = clip_boxes_to_image((bbox + 0.5).long(), (H, W))

            # output mask
            sample['mask_bbox'] = torch.zeros((1, H, W), dtype=torch.float32)
            for x1, y1, x2, y2 in bbox.tolist():
                sample['mask_bbox'][:, y1:y2+1, x1:x2+1] = 1.0

        return sample


class RegionDescriptionsVisualGnome(RECDataset):
    def __init__(self, data_root, transform=None, tokenizer=None,
                 max_length=32, with_mask_bbox=False):
        super().__init__(transform=transform, tokenizer=tokenizer,
                         max_length=max_length, with_mask_bbox=with_mask_bbox)


        # if available, read COCO IDs from the val, testA and testB splits from
        # the RefCOCO dataset
        try:
            with open('./refcoco_valtest_ids.txt', 'r') as fh:
                refcoco_ids = [int(lin.strip()) for lin in fh.readlines()]
        except:
            refcoco_ids = []

        def path_from_url(fname):
            return os.path.join(data_root, fname[fname.index('VG_100K'):])

        with open(os.path.join(data_root, 'image_data.json'), 'r') as f:
            image_data = {
                data['image_id']: path_from_url(data['url'])
                for data in json.load(f)
                if data['coco_id'] is None or data['coco_id'] not in refcoco_ids
            }
        print(f'{len(image_data)} images')

        self.samples = []

        with open(os.path.join(data_root, 'region_descriptions.json'), 'r') as f:
            for record in progressbar(ijson.items(f, 'item.regions.item'), desc='loading data'):
                if record['image_id'] not in image_data:
                    continue
                file_name = image_data[record['image_id']]

                expr = record['phrase']

                bbox = [record['x'], record['y'], record['width'], record['height']]
                bbox = torch.atleast_2d(torch.FloatTensor(bbox))
                bbox = box_convert(bbox, 'xywh', 'xyxy')  # xyxy

                self.samples.append((file_name, expr, bbox))

        self.print_stats()


class ReferDataset(RECDataset):
    def __init__(self, data_root, dataset, split_by, split, transform=None,
                 tokenizer=None, max_length=32, with_mask_bbox=False):
        super().__init__(transform=transform, tokenizer=tokenizer,
                         max_length=max_length, with_mask_bbox=with_mask_bbox)

        # https://github.com/lichengunc/refer
        try:
            import sys
            sys.path.append('refer')
            from refer import REFER
        except:
            raise RuntimeError('create a symlink to valid refer compilation '
                               '(see https://github.com/lichengunc/refer)')

        refer = REFER(data_root, dataset, split_by)
        ref_ids = sorted(refer.getRefIds(split=split))

        self.samples = []

        for rid in progressbar(ref_ids, desc='loading data'):
            ref = refer.Refs[rid]
            ann = refer.refToAnn[rid]

            file_name = refer.Imgs[ref['image_id']]['file_name']
            if dataset == 'refclef':
                file_name = os.path.join(
                    'refer', 'data', 'images', 'saiapr_tc-12', file_name
                )
            else:
                coco_set = file_name.split('_')[1]
                file_name = os.path.join(
                    'refer', 'data', 'images', 'mscoco', coco_set, file_name
                )

            bbox = ann['bbox']
            bbox = torch.atleast_2d(torch.FloatTensor(bbox))
            bbox = box_convert(bbox, 'xywh', 'xyxy')  # xyxy

            sentences = [s['sent'] for s in ref['sentences']]
            if 'train' in split:  # remove repeated expresions
                sentences = list(set(sentences))
            sentences = sorted(sentences)

            self.samples += [(file_name, expr, bbox) for expr in sentences]

        self.print_stats()


class RefCLEF(ReferDataset):
    def __init__(self, *args, **kwargs):
        assert args[0] in ('train', 'val', 'test')
        super().__init__('refer/data', 'refclef', 'berkeley', *args, **kwargs)


class RefCOCO(ReferDataset):
    def __init__(self, *args, **kwargs):
        assert args[0] in ('train', 'val', 'trainval', 'testA', 'testB')
        super().__init__('refer/data', 'refcoco', 'unc', *args, **kwargs)


class RefCOCOp(ReferDataset):
    def __init__(self, *args, **kwargs):
        assert args[0] in ('train', 'val', 'trainval', 'testA', 'testB')
        super().__init__('refer/data', 'refcoco+', 'unc', *args, **kwargs)


class RefCOCOg(ReferDataset):
    def __init__(self, *args, **kwargs):
        assert args[0] in ('train', 'val', 'test')
        super().__init__('refer/data', 'refcocog', 'umd', *args, **kwargs)
