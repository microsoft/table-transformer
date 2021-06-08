import random
import torch
import math
import PIL
from PIL import ImageFilter

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
    
class RandomCrop(object):
    def __init__(self, prob, left_scale, top_scale, right_scale, bottom_scale):
        self.prob = prob
        self.left_scale = left_scale
        self.top_scale = top_scale
        self.right_scale = right_scale
        self.bottom_scale = bottom_scale

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
                bbox = [max(bbox[0], left) - left,
                        max(bbox[1], top) - top,
                        min(bbox[2], right) - left,
                        min(bbox[3], bottom) - top]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    cropped_labels.append(label)
                         
            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
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
    def __init__(self, prob, min_scale_factor, max_scale_factor):
        self.prob = prob
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def __call__(self, image, target):
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
                target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(resized_labels, dtype=torch.int64)
                return resized_image, target

        return image, target
    

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
