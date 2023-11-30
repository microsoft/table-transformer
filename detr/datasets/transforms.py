# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
from util import box_ops

def compute_area(boxes):
    return (boxes[:, 1, :] - boxes[:, 0, :]).prod(dim=1)

def compute_area_with_bounds_and_presence(boxes, present_inside, present_outside):
    area = compute_area(boxes)
    area[~present_inside] = 0
    area_outside = (boxes[:, 3, :] - boxes[:, 2, :]).prod(dim=1)
    area_outside[~present_outside] = 0
    return torch.max(area, area_outside)

def crop(image, original_target, region, enable_bounds):
    cropped_image = F.crop(image, *region)

    target = original_target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        subtrahend = torch.as_tensor([j, i, j, i])
        if enable_bounds:
            # Present/Empty status remains unchanged.
            subtrahend = torch.cat((subtrahend, subtrahend))
        cropped_boxes = boxes - subtrahend
        if enable_bounds:
            present_inside = box_ops.is_present(cropped_boxes[:, :4])
            present_outside = box_ops.is_present(cropped_boxes[:, 4:])
        # Next operations can make a non-present box present again.
        cropped_boxes = torch.min(cropped_boxes.reshape(
            -1, 4 if enable_bounds else 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        reshaped_cropped_boxes = cropped_boxes.reshape(-1, 8 if enable_bounds else 4)
        if enable_bounds:
            # Rectify non-present boxes.
            reshaped_cropped_boxes[:, :4][box_ops.is_present(reshaped_cropped_boxes[:, :4]) & ~present_inside] = box_ops.MISSING_BOX
            reshaped_cropped_boxes[:, 4:][box_ops.is_present(reshaped_cropped_boxes[:, 4:]) & ~present_outside] = box_ops.MISSING_BOX
        target["boxes"] = reshaped_cropped_boxes
        target["area"] = compute_area_with_bounds_and_presence(
            cropped_boxes, present_inside, present_outside) if enable_bounds else compute_area(cropped_boxes)
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target and len(target["boxes"]) > 0:
            cropped_boxes = target['boxes'].reshape(-1, 4 if enable_bounds else 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            if enable_bounds:
                keep += torch.all(cropped_boxes[:, 3, :] > cropped_boxes[:, 2, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target, enable_bounds):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"]
        indices = [2, 1, 0, 3]
        if enable_bounds:
            indices += [4 + x for x in indices]
        factor = torch.as_tensor([-1, 1, -1, 1])
        if enable_bounds:
            factor = torch.cat((factor, factor))
        offset = torch.as_tensor([w, 0, w, 0])
        if enable_bounds:
            offset = torch.cat((offset, offset))
        boxes = boxes[:, indices] * factor + offset
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size, enable_bounds):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"]
        factor = torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        if enable_bounds:
            factor = torch.cat(factor, factor)
        scaled_boxes = boxes * factor
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size, enable_bounds):
        self.size = size
        self.enable_bounds = enable_bounds

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region, self.enable_bounds)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, enable_bounds: bool):
        self.min_size = min_size
        self.max_size = max_size
        self.enable_bounds = enable_bounds

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region, self.enable_bounds)


class CenterCrop(object):
    def __init__(self, size, enable_bounds):
        self.size = size
        self.enable_bounds = enable_bounds

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width), self.enable_bounds)


class RandomHorizontalFlip(object):
    def __init__(self, p, enable_bounds):
        self.p = p
        self.enable_bounds = enable_bounds

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target, self.enable_bounds)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size, enable_bounds):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.enable_bounds = enable_bounds

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, self.enable_bounds)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            image, tgt = self.transforms1(img, target)
            return image, tgt
        image, tgt = self.transforms2(img, target)
        return image, tgt

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std, enable_bounds):
        self.mean = mean
        self.std = std
        self.enable_bounds = enable_bounds

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if self.enable_bounds:
                boxes = torch.cat((box_xyxy_to_cxcywh(boxes[:, :4]), box_xyxy_to_cxcywh(boxes[:, 4:])), dim=1)
                boxes = boxes / torch.tensor([w, h, w, h] * 2, dtype=torch.float32)
            else:
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
