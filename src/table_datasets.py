"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import os.path
import sys
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import itertools
import math

import PIL
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import re
import collections

# Project imports
sys.path.append("detr")
import datasets.transforms as R
from util import box_ops

_PATTERN = re.compile(r"(.*) (\S+) ([io])")

def convert_to_bounds(label_bbox_pairs):
    default_missing_box = box_ops.MISSING_BOX.tolist()
    d = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: [default_missing_box, default_missing_box]))
    normal_pairs = []
    for label, bbox in label_bbox_pairs:
        m = _PATTERN.fullmatch(label)
        if not m:
            normal_pairs.append((label, bbox * 2))
            continue
        assert m.group(3) in ["i", "o"]
        outside = m.group(3) == "o"
        io = d[m.group(1)][m.group(2)]
        assert io[outside] == default_missing_box
        io[outside] = bbox
    pairs = []
    for label, id2io in d.items():
        for _, io in id2io.items():
            assert len(io) == 2
            assert io[0]
            assert io[1]
            pairs.append((label, sum(io, [])))
    pairs.extend(normal_pairs)
    return pairs
assert(
    convert_to_bounds([("row", [1, 2, 3, 4])]) == [('row', [1, 2, 3, 4, 1, 2, 3, 4])])
assert(
    convert_to_bounds([("row f o", [5, 6, 7, 8]), ("row f i", [11, 12, 13, 14])]) == [('row', [11, 12, 13, 14, 5, 6, 7, 8])])
assert(
    convert_to_bounds([("row f o", [5, 6, 7, 8])]) == [('row', [5, 5, 4, 5, 5, 6, 7, 8])]
)

def read_pascal_voc(xml_file: str, class_map, enable_bounds):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    pairs = []
    for object_ in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        
        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        pairs.append((label, bbox))
        
    if enable_bounds:
        pairs = convert_to_bounds(pairs)
        
    bboxes = []
    labels = []
    for label, bbox in pairs:
        try:
            label = int(label)
        except:
            label = int(class_map[label])
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels

# def crop_around_bbox_coco(image, crop_bbox, max_margin, target):
#     width, height = image.size
#     left = max(1, int(round(crop_bbox[0] - max_margin * random.random())))
#     top = max(1, int(round(crop_bbox[1] - max_margin * random.random())))
#     right = min(width, int(round(crop_bbox[2] + max_margin * random.random())))
#     bottom = min(height, int(round(crop_bbox[3] + max_margin * random.random())))
#     cropped_image = image.crop((left, top, right, bottom))
#     cropped_bboxes = []
#     cropped_labels = []
#     for bbox, label in zip(target["boxes"], target["labels"]):
#         bbox = list_bbox_cxcywh_to_xyxy(bbox)
#         bbox = [max(bbox[0], left) - left,
#                 max(bbox[1], top) - top,
#                 min(bbox[2], right) - left,
#                 min(bbox[3], bottom) - top]
#         if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
#             bbox = list_bbox_xyxy_to_cxcywh(bbox)
#             cropped_bboxes.append(bbox)
#             cropped_labels.append(label)

#     if len(cropped_bboxes) > 0:
#         target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
#         target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
#         w, h = img.size
#         target["size"] = torch.tensor([w, h])
#         return cropped_image, target
                 
#     return image, target


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)


# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob, enable_bounds):
        self.prob = prob
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            if self.enable_bounds:
                bbox[:, [4, 6]] = width - bbox[:, [6, 4]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
    
class RandomCrop(object):
    def __init__(self, prob, left_scale, top_scale, right_scale, bottom_scale, enable_bounds):
        assert False
        self.prob = prob
        self.left_scale = left_scale
        self.top_scale = top_scale
        self.right_scale = right_scale
        self.bottom_scale = bottom_scale
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        assert False
        if random.random() < self.prob:
            width, height = image.size
            left = int(math.floor(width * 0.5 * self.left_scale * random.random()))
            top = int(math.floor(height * 0.5 * self.top_scale * random.random()))
            right = width - int(math.floor(width * 0.5 * self.right_scale * random.random()))
            bottom = height - int(math.floor(height * 0.5 * self.bottom_scale * random.random()))
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                bbox = [max(bbox[0], left) - left,
                        max(bbox[1], top) - top,
                        min(bbox[2], right) - left,
                        min(bbox[3], bottom) - top]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)
                         
            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target
    
    
class RandomBlur(object):
    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius

    def __call__(self, image, target):
        if random.random() < self.prob:
            radius = random.random() * self.max_radius
            image = image.filter(filter=ImageFilter.GaussianBlur(radius=radius))

        return image, target
    
    
class RandomResize(object):
    def __init__(self, prob, min_scale_factor, max_scale_factor, enable_bounds):
        assert False
        self.prob = prob
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        assert False
        if random.random() < self.prob:
            prob = random.random()
            scale_factor = prob*self.max_scale_factor + (1-prob)*self.min_scale_factor
            new_width = int(round(scale_factor * image.width))
            new_height = int(round(scale_factor * image.height))
            resized_image = image.resize((new_width, new_height), resample=PIL.Image.LANCZOS)
            resized_bboxes = []
            resized_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                bbox = [elem*scale_factor for elem in bbox]
                if bbox[0] < bbox[2] - 1 and bbox[1] < bbox[3] - 1:
                    resized_bboxes.append(bbox)
                    resized_labels.append(label)
                         
            if len(resized_bboxes) > 0:
                target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
                target["labels"] = torch.as_tensor(resized_labels, dtype=torch.int64)
                return resized_image, target

        return image, target
    

# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, image, target=None):
#         image = F.normalize(image, mean=self.mean, std=self.std)
#         if target is None:
#             return image, None
#         target = target.copy()
#         h, w = image.shape[-2:]
#         if "boxes" in target:
#             boxes = target["boxes"]
#             boxes = box_xyxy_to_cxcywh(boxes)
#             boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
#             target["boxes"] = boxes
#         return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class TightAnnotationCrop(object):
    def __init__(self, labels, left_max_pad, top_max_pad, right_max_pad, bottom_max_pad, enable_bounds):
        self.labels = set(labels)
        self.left_max_pad = left_max_pad
        self.top_max_pad = top_max_pad
        self.right_max_pad = right_max_pad
        self.bottom_max_pad = bottom_max_pad
        self.enable_bounds = enable_bounds

    def __call__(self, img: PIL.Image.Image, target: dict):
        w, h = target['size']
        bboxes = [bbox for label, bbox in zip(target['labels'], target['boxes']) if label.item() in self.labels]
        if len(bboxes) > 0:
            object_num = random.randint(0, len(bboxes)-1)
            left = random.randint(0, self.left_max_pad)
            top = random.randint(0, self.top_max_pad)
            right = random.randint(0, self.right_max_pad)
            bottom = random.randint(0, self.bottom_max_pad)
            bbox_tensor = bboxes[object_num]
            #target["crop_orig_size"] = torch.tensor([bbox[3]-bbox[1]+y_margin*2, bbox[2]-bbox[0]+x_margin*2])
            #target["crop_orig_offset"] = torch.tensor([bbox[0]-x_margin, bbox[1]-y_margin])
            if self.enable_bounds:
                assert bbox_tensor.shape == (8, )
                present = []
                if box_ops.is_present(bbox_tensor[:4]).item():
                    present.append(bbox_tensor[:4])
                # Probably we do not need the inner boundary here but it seems safer to include it.
                if box_ops.is_present(bbox_tensor[4:]).item():
                    present.append(bbox_tensor[4:])
                if not box_ops:
                    return img, target
                stack = torch.stack(present)
                bbox = torch.cat((stack[:, :2].min(dim=0)[0], stack[:, 2:].max(dim=0)[0]))
                if bbox.isfinite().all():
                    bbox = bbox.tolist()
                else:
                    return img, target
            else:
                bbox = bbox_tensor.tolist()
            region = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            # transpose and add margin
            region = [region[1]-top, region[0]-left, region[3]+top+bottom, region[2]+left+right]
            # print("region: {}".format(region))
            region = [round(elem) for elem in region]
            return R.crop(img, target, region, self.enable_bounds)
        else:
            return img, target

class RandomCrop(object):
    def __init__(self, prob, left_pixels, top_pixels, right_pixels, bottom_pixels, enable_bounds):
        assert False
        self.prob = prob
        self.left_pixels= left_pixels
        self.top_pixels = top_pixels
        self.right_pixels = right_pixels
        self.bottom_pixels = bottom_pixels
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        assert False
        if random.random() < self.prob:
            width, height = image.size
            left = random.randint(0, self.left_pixels)
            top = random.randint(0, self.top_pixels)
            right = width - random.randint(0, self.right_pixels)
            bottom = height - random.randint(0, self.bottom_pixels)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                bbox = [max(bbox[0], left) - left,
                        max(bbox[1], top) - top,
                        min(bbox[2], right) - left,
                        min(bbox[3], bottom) - top]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)
                         
            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target

class RandomPercentageCrop(object):
    def __init__(self, prob, left_scale, top_scale, right_scale, bottom_scale, enable_bounds):
        self.prob = prob
        self.left_scale = left_scale
        self.top_scale = top_scale
        self.right_scale = right_scale
        self.bottom_scale = bottom_scale
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            left = int(math.floor(width * 0.5 * self.left_scale * random.random()))
            top = int(math.floor(height * 0.5 * self.top_scale * random.random()))
            right = width - int(math.floor(width * 0.5 * self.right_scale * random.random()))
            bottom = height - int(math.floor(height * 0.5 * self.bottom_scale * random.random()))
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            cropped_labels = []
            for bbox, label in zip(target["boxes"], target["labels"]):
                # assert len(bbox.shape) == 2, bbox.shape
                # assert bbox.shape[1] == 4
                bbox_inside = [max(bbox[0], left) - left,
                        max(bbox[1], top) - top,
                        min(bbox[2], right) - left,
                        min(bbox[3], bottom) - top]
                # assert bbox_inside[0] <= bbox_inside[2], bbox_inside
                # assert bbox_inside[1] <= bbox_inside[3], bbox_inside
                if self.enable_bounds:
                    bbox_outside = [max(bbox[4], left) - left,
                        max(bbox[5], top) - top,
                        min(bbox[6], right) - left,
                        min(bbox[7], bottom) - top]
                if (bbox_inside[0] < bbox_inside[2] and bbox_inside[1] < bbox_inside[3]) or (
                    self.enable_bounds and bbox_outside[0] < bbox_outside[2] and bbox_outside[1] < bbox_outside[3]
                ):
                    b = bbox_inside
                    if self.enable_bounds:
                        b += bbox_outside
                    cropped_bboxes.append(b)
                    cropped_labels.append(label)

            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
                target["labels"] = torch.as_tensor(cropped_labels, dtype=torch.int64)
                return cropped_image, target

        return image, target

class ColorJitterWithTarget(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = transforms.ColorJitter(brightness=brightness,
                                                contrast=contrast,
                                                saturation=saturation,
                                                hue=hue)

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target

class RandomErasingWithTarget(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=255, inplace=False):
        self.transform = transforms.RandomErasing(p=p,
                                                  scale=scale,
                                                  ratio=ratio,
                                                  value=value,
                                                  inplace=False)

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"

class ToPILImageWithTarget(object):
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, img: PIL.Image.Image, target: dict):
        img = self.transform(img)

        return img, target
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"

class RandomDilation(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(size, int(round(0 * size * size))) # 0 is equivalent to a min filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        
        if r <= self.probability:
            img = img.filter(self.filter)
        
        return img, target

class RandomErosion(object):
    def __init__(self, probability=0.5, size=3):
        self.probability = probability
        self.filter = ImageFilter.RankFilter(size, int(round(0.6 * size * size))) # Almost a median filter

    def __call__(self, img: PIL.Image.Image, target: dict):
        r = random.random()
        
        if r <= self.probability:
            img = img.filter(self.filter)
        
        return img, target

class RandomResize(object):
    def __init__(self, min_min_size, max_min_size, max_max_size, enable_bounds):
        self.min_min_size = min_min_size
        self.max_min_size = max_min_size
        self.max_max_size = max_max_size
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        width, height = image.size
        current_min_size = min(width, height)
        current_max_size = max(width, height)
        min_size = random.randint(self.min_min_size, self.max_min_size)
        if current_max_size * min_size / current_min_size > self.max_max_size:
            scale = self.max_max_size / current_max_size
        else:
            scale = min_size / current_min_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale*elem for elem in bbox]
            assert len(bbox) == (8 if self.enable_bounds else 4)
            resized_bboxes.append(bbox)

        target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
        
        return resized_image, target

class RandomMaxResize(object):
    def __init__(self, min_max_size, max_max_size, enable_bounds):
        self.min_max_size = min_max_size
        self.max_max_size = max_max_size
        self.enable_bounds = enable_bounds

    def __call__(self, image, target):
        width, height = image.size
        current_max_size = max(width, height)
        target_max_size = random.randint(self.min_max_size, self.max_max_size)
        scale = target_max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale*elem for elem in bbox]
            resized_bboxes.append(bbox)

        target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32).reshape(-1, 8 if self.enable_bounds else 4)
        
        return resized_image, target

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"

def normalize(enable_bounds):
    return R.Compose([
        R.ToTensor(),
        R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], enable_bounds)
        ])

random_erasing = R.Compose([
    R.ToTensor(),
    RandomErasingWithTarget(p=0.5,
                            scale=(0.003, 0.03),
                            ratio=(0.1, 0.3),
                            value='random'),
    RandomErasingWithTarget(p=0.5,
                            scale=(0.003, 0.03),
                            ratio=(0.3, 1),
                            value='random'),
    ToPILImageWithTarget()
])


def get_structure_transform(image_set, enable_bounds):
    """
    returns the appropriate transforms for structure recognition.
    """

    if image_set == 'train':
        return R.Compose([
            R.RandomSelect(TightAnnotationCrop([0], 30, 30, 30, 30, enable_bounds),
                           TightAnnotationCrop([0], 10, 10, 10, 10, enable_bounds),
                           p=0.5),
            RandomMaxResize(900, 1100, enable_bounds), random_erasing, normalize(enable_bounds)
        ])

    if image_set == 'val':
        return R.Compose([RandomMaxResize(1000, 1000, enable_bounds), normalize(enable_bounds)])

    raise ValueError(f'unknown {image_set}')


def get_detection_transform(image_set, enable_bounds):
    """
    returns the appropriate transforms for table detection.
    """

    if image_set == 'train':
        return R.Compose([
            R.RandomSelect(TightAnnotationCrop([0, 1], 100, 150, 100, 150, enable_bounds),
                           RandomPercentageCrop(1, 0.1, 0.1, 0.1, 0.1, enable_bounds),
                           p=0.2),
            RandomMaxResize(704, 896, enable_bounds), normalize(enable_bounds)
        ])

    if image_set == 'val':
        return R.Compose([RandomMaxResize(800, 800, enable_bounds), normalize(enable_bounds)])

    raise ValueError(f'unknown {image_set}')


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def compute_area(bboxes):
    # Why does the original code not do the subtraction_
    # bboxes[:, 2] * bboxes[:, 3] # COCO area
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

def compute_area_with_bounds(bboxes):
    # print("bboxes: {}".format(bboxes))
    area_inside = compute_area(bboxes[:, :4])
    present_inside = box_ops.is_present(bboxes[:, :4])
    area_inside[~present_inside] = 0
    area_outside = compute_area(bboxes[:, 4:])
    present_outside = box_ops.is_present(bboxes[:, 4:])
    area_outside[~present_outside] = 0
    # print("area_inside: {}".format(area_inside))
    # print("area_outside: {}".format(area_outside))
    # print("torch.stack((area_inside, area_outside)): {}".format(
    #     torch.stack((area_inside, area_outside))))
    # print("torch.max(torch.stack((area_inside, area_outside)), dim=0): {}".format(
    #     torch.max(torch.stack((area_inside, area_outside)), dim=0)
    # ))
    return torch.max(torch.stack((area_inside, area_outside)), dim=0)

def box_xyxy_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


class PDFTablesDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, max_size=None, do_crop=True, make_coco=False,
                 include_eval=False, max_neg=None, negatives_root=None, xml_fileset="filelist.txt",
                image_extension='.png', class_map=None, enable_bounds=False):
        self.root = root
        self.transforms = transforms
        self.do_crop=do_crop
        self.make_coco = make_coco
        self.image_extension = image_extension
        self.include_eval = include_eval
        self.class_map = class_map
        self.class_list = list(class_map)
        self.class_set = set(class_map.values())
        self.class_set.remove(class_map['no object'])
        self.enable_bounds = enable_bounds

        print(os.path.join(os.path.dirname(root), xml_fileset))
        try:
            with open(os.path.join(os.path.dirname(root), xml_fileset), 'r') as file:
                lines = file.readlines()
                lines = [l.split('/')[-1] for l in lines]
        except:
            lines = os.listdir(root)
        xml_page_ids = set([f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")])
            
        image_directory = os.path.join(os.path.dirname(root), "images")
        try:
            with open(os.path.join(image_directory, "filelist.txt"), 'r') as file:
                lines = file.readlines()
        except:
            lines = os.listdir(image_directory)
        png_page_ids = {root for root, ext in (os.path.splitext(f.strip()) for f in lines) if ext == self.image_extension}
        
        self.page_ids = sorted(xml_page_ids.intersection(png_page_ids))
        if not max_size is None:
            random.shuffle(self.page_ids)
            self.page_ids = self.page_ids[:max_size]
        num_page_ids = len(self.page_ids)
        self.types = [1 for idx in range(num_page_ids)]
            
        if not max_neg is None and max_neg > 0:
            with open(os.path.join(negatives_root, "filelist.txt"), 'r') as file:
                neg_xml_page_ids = set([f.strip().replace(".xml", "") for f in file.readlines() if f.strip().endswith(".xml")])
                neg_xml_page_ids = neg_xml_page_ids.intersection(png_page_ids)
                neg_xml_page_ids = sorted(neg_xml_page_ids.difference(set(self.page_ids)))
                if len(neg_xml_page_ids) > max_neg:
                    neg_xml_page_ids = neg_xml_page_ids[:max_neg]
            self.page_ids += neg_xml_page_ids
            self.types += [0 for idx in range(len(neg_xml_page_ids))]
        
        self.has_mask = False
        
        if self.make_coco:
            self.dataset = {}
            self.dataset['images'] = [{'id': idx} for idx, _ in enumerate(self.page_ids)]
            self.dataset['annotations'] = []
            ann_id = 0
            for image_id, page_id in enumerate(self.page_ids):
                annot_path = os.path.join(self.root, page_id + ".xml")
                bboxes, labels = read_pascal_voc(annot_path, class_map=self.class_map, enable_bounds=self.enable_bounds)

                # Reduce class set
                keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
                bboxes = [bboxes[idx] for idx in keep_indices]
                labels = [labels[idx] for idx in keep_indices]

                torch_boxes = (
                    torch.as_tensor(data=bboxes, dtype=torch.float32) if bboxes
                    else torch.empty((0, 8 if enable_bounds else 4))
                )
                if self.enable_bounds:
                    areas, is_outside = compute_area_with_bounds(torch_boxes)
                else:
                    areas = compute_area(torch_boxes)

                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    ann = {'area': areas[i].item(),
                           'iscrowd': 0,
                           'bbox': box_xyxy_to_xywh(bbox[is_outside[i].item() * 4:][:4] if enable_bounds else bbox),
                           'category_id': label,
                           'image_id': image_id,
                           'id': ann_id,
                           'ignore': 0,
                           'segmentation': []}
                    self.dataset['annotations'].append(ann)
                    ann_id += 1
            self.dataset['categories'] = [{'id': idx} for idx in self.class_list[:-1]]

            self.createIndex()
            
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def __getitem__(self, idx):
        # load images ad masks
        page_id = self.page_ids[idx]
        img_path = os.path.join(os.path.dirname(self.root), "images", page_id + self.image_extension)
        annot_path = os.path.join(self.root, page_id + ".xml")
        
        try:
            img = Image.open(img_path)
        except os.error as e:
            print("Error reading file: {}".format(img_path))
            raise e
        try:
            img = img.convert("RGB")
        except os.error as e:
            print("Error converting file: {}".format(img_path))
            raise e
        
        w, h = img.size
        
        if self.types[idx] == 1:
            bboxes, labels = read_pascal_voc(annot_path, class_map=self.class_map, enable_bounds=self.enable_bounds)

            # Reduce class set
            keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
            bboxes = [bboxes[idx] for idx in keep_indices]
            labels = [labels[idx] for idx in keep_indices]

            # Convert to Torch Tensor
            if len(labels) > 0:
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            else:
                # Not clear if it's necessary to force the shape of bboxes to be (0, 4)
                bboxes = torch.empty((0, 8 if self.enable_bounds else 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 8 if self.enable_bounds else 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        num_objs = bboxes.shape[0]

        # Create target
        target = {}
        assert(bboxes.shape[-1] == (8 if self.enable_bounds else 4)), bboxes.shape
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = compute_area_with_bounds(bboxes)[0] if self.enable_bounds else compute_area(bboxes)
        # Warning: why did the original code ignore the top-left corner?
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.include_eval:
            target["img_path"] = img_path

        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)
        
        #if self.include_original:
        #    return img_tensor, target, img, img_path

        return img_tensor, target

    def __len__(self):
        return len(self.page_ids)
    
    def getImgIds(self):
        return range(len(self.page_ids))
    
    def getCatIds(self):
        return range(10)
    
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
    
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

            ids = [ann['id'] for ann in anns]
        return ids
