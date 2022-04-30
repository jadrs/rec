import torch

from torchvision import transforms

from torchvision.transforms import Compose

from PIL import Image


class ToTensor(transforms.ToTensor):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)
        assert 'image' in input
        input['image'] = super().__call__(input['image'])
        return input


class Normalize(transforms.Normalize):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)
        assert 'image' in input
        input['image'] = super().__call__(input['image'])
        return input


class NormalizeBoxCoords(transforms.ToTensor):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)
        assert 'image' in input and 'bbox' in input
        _, H, W = input['image'].size()
        input['bbox'][:, (0, 2)] /= W
        input['bbox'][:, (1, 3)] /= H

        if 'tr_param' not in input:
            input['tr_param'] = []
        input['tr_param'].append({'normalize_box_coords': (H, W)})

        return input


class SquarePad(torch.nn.Module):
    def __call__(self, input):
        if isinstance(input, Image.Image):
            raise NotImplementedError('put the SquarePad transform after ToTensor')

        assert 'image' in input
        _, h, w = input['image'].size()

        max_wh = max(w, h)
        xp = int(0.5 * (max_wh - w))
        yp = int(0.5 * (max_wh - h))
        padding = (xp, yp, (max_wh-xp)-w, (max_wh-yp)-h)

        input['image'] = transforms.functional.pad(
            input['image'], padding, fill=0, padding_mode='constant'
        )
        # input['image'] = transforms.functional.pad(
        #     input['image'], padding, padding_mode='edge'
        # )

        if 'mask' in input:
            input['mask'] = transforms.functional.pad(
                input['mask'], padding, fill=0, padding_mode='constant'
            )

        if 'bbox' in input:
            input['bbox'][:, (0, 2)] += xp
            input['bbox'][:, (1, 3)] += yp

        if 'tr_param' not in input:
            input['tr_param'] = []
        input['tr_param'].append({'square_pad': padding})

        return input


class Resize(transforms.Resize):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)

        assert 'image' in input

        if not torch.is_tensor(input['image']):
            raise NotImplementedError('put the Resize transform after ToTensor')

        _, img_h, img_w = input['image'].size()

        if isinstance(self.size, int):
            dst_h = self.size if img_h < img_w else int(self.size * img_h / img_w)
            dst_w = self.size if img_w < img_h else int(self.size * img_w / img_h)
        else:
            dst_h, dst_w = self.size

        input['image'] = super().__call__(input['image'])

        if 'mask' in input:
            input['mask'] = super().__call__(input['mask'])

        sx, sy = dst_w / img_w, dst_h / img_h

        if 'bbox' in input:
            input['bbox'][:, (0, 2)] *= sx
            input['bbox'][:, (1, 3)] *= sy

        if 'tr_param' not in input:
            input['tr_param'] = []
        input['tr_param'].append({'resize': (sx, sy)})

        return input


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)

        assert 'image' in input

        if not torch.is_tensor(input['image']):
            raise NotImplementedError('use Resize after ToTensor')

        result = super().__call__(input['image'])
        if result is input['image']:  # not flipped
            return input
        input['image'] = result

        if 'mask' in input:
            input['mask'] = torch.flip(input['mask'], dims=(-1,))

        img_w = input['image'].size(2)

        if 'bbox' in input:
            input['bbox'][:, (0, 2)] = img_w - input['bbox'][:, (2, 0)]

        if 'expr' in input:
            input['expr'] = input['expr'].replace('left', '<LEFT>').replace('right', 'left').replace('<LEFT>', 'right')

        # if 'tr_param' not in input:
        #     input['tr_param'] = []
        # input['tr_param'].append({'random_horizontal_flip': img_w})

        return input


class RandomAffine(transforms.RandomAffine):
    def get_params(self, *args, **kwargs):
        self.params = super().get_params(*args, **kwargs)
        return self.params

    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)

        assert 'image' in input

        if not torch.is_tensor(input['image']):
            raise NotImplementedError('put the Resize transform after ToTensor')

        #self.fill = input['image'].mean((1,2))  # set fill value to the mean pixel value
        result = super().__call__(input['image'])
        if result is input['image']:  # not transformed
            return input
        input['image'] = result

        _, img_h, img_w = input['image'].size()

        angle, translate, scale, shear = self.params
        center = (img_w * 0.5, img_h * 0.5)
        matrix = transforms.functional._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        matrix = torch.FloatTensor([matrix[:3], matrix[3:], [0, 0, 1]])
        matrix = torch.linalg.inv(matrix)

        if 'mask' in input:
            input['mask'] = transforms.functional.affine(
                input['mask'], *self.params, self.interpolation, self.fill
            )

        if 'bbox' in input:
            for i, (x1, y1, x2, y2) in enumerate(input['bbox']):
                pt = matrix @ torch.FloatTensor([
                    [x1, y1, 1],
                    [x2, y1, 1],
                    [x2, y2, 1],
                    [x1, y2, 1]
                ]).T
                x_min, y_min, _ = pt.min(dim=1).values
                x_max, y_max, _ = pt.max(dim=1).values
                input['bbox'][i, :] = torch.FloatTensor([x_min, y_min, x_max, y_max])

        # if 'tr_param' not in input:
        #     input['tr_param'] = []
        # input['tr_param'].append({'random_affine': matrix[:2, :].tolist()})

        return input


class ColorJitter(transforms.ColorJitter):
    def __call__(self, input):
        if not isinstance(input, dict):
            return super().__call__(input)
        assert 'image' in input
        input['image'] = super().__call__(input['image'])
        return input


def get_transform(split, input_size=512):
    mean = [0.485, 0.456, 0.406]
    sdev = [0.229, 0.224, 0.225]

    if split in ('train', 'trainval'):
        transform = Compose([
            # ColorJitter(brightness=0.5, saturation=0.5),  # before normalization
            ToTensor(),
            Normalize(mean, sdev),  # first normalize so that the mean is ~0
            SquarePad(),  # zero pad (approx mean pixel value)
            Resize(size=(input_size, input_size)),
            # RandomHorizontalFlip(p=0.5),
            RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            NormalizeBoxCoords(),
        ])
    elif split in ('val', 'test', 'testA', 'testB', 'testC'):
        transform = Compose([
            ToTensor(),
            Normalize(mean, sdev),
            SquarePad(),
            Resize(size=(input_size, input_size)),
            NormalizeBoxCoords(),
        ])
    elif split in ('visu',):
        transform = Compose([
            ToTensor(),
            SquarePad(),
            Resize(size=(input_size, input_size)),
            NormalizeBoxCoords(),
        ])
    else:
        raise ValueError(f'\'{split}\' is not a valid data split')

    return transform


def denormalize(img):
    mean = [0.485, 0.456, 0.406]
    sdev = [0.229, 0.224, 0.225]
    return Normalize(
        mean=[-m/s for m, s in zip(mean, sdev)], std=[1./s for s in sdev]
    )(img)


def undo_box_transforms(bbox, tr_param):
    # undo validation mode transformations
    bbox = bbox.clone()
    for tr in tr_param[::-1]:
        if 'resize' in tr:
            sx, sy = tr['resize']
            bbox[:, (0, 2)] /= sx
            bbox[:, (1, 3)] /= sy
        elif 'square_pad' in tr:
            px, py, _, _ = tr['square_pad']
            bbox[:, (0, 2)] -= px
            bbox[:, (1, 3)] -= py
        elif 'normalize_box_coords' in tr:
            img_h, img_w = tr['normalize_box_coords']
            bbox[:, (0, 2)] *= img_w
            bbox[:, (1, 3)] *= img_h
        else:
            continue
    return bbox


def undo_box_transforms_batch(bbox, tr_param):
    output = []
    for i in range(bbox.size(0)):
        bb = undo_box_transforms(torch.atleast_2d(bbox[i]), tr_param[i])
        output.append(bb)
    return torch.cat(output, dim=0)
