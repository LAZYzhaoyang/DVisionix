# -*- coding: utf-8 -*-
"""几何同步变换。

任务无关，但会同步处理 ``boxes`` (xyxy) 与 ``mask``（如果存在），
因此分类 / 检测 / 分割都可以复用同一套。

约定：boxes 是 numpy float32 (N, 4) [x1, y1, x2, y2] 绝对坐标；
mask 是 numpy int64 (H, W)。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..sample import Sample
from .base import BaseTransform
from ...registry import TRANSFORMS


def _filter_invalid_boxes(boxes: np.ndarray, labels: Optional[np.ndarray]):
    """过滤退化的 box（w<=0 或 h<=0），并同步裁剪 labels。"""
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if labels is not None and len(labels) == len(valid):
        labels = labels[valid]
    return boxes, labels


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_resize")
class BoxSyncResize(BaseTransform):
    """resize image + 同步缩放 boxes（xyxy）+ mask。"""

    name = "box_sync_resize"

    def __init__(self, size: Optional[Tuple[int, int]] = None, max_size: Optional[int] = None):
        if size is None and max_size is None:
            raise ValueError("Either size or max_size must be specified.")
        self.size, self.max_size = size, max_size

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        h, w = img.shape[:2]
        if self.size is not None:
            th, tw = self.size
            sh, sw = th / h, tw / w
        else:
            s = self.max_size / max(h, w)
            sh = sw = s
            th, tw = int(round(h * s)), int(round(w * s))
        sample["image"] = cv2.resize(img, (tw, th))

        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] *= sw
            boxes[:, [1, 3]] *= sh
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th)
            labels = sample.get("labels")
            boxes, labels = _filter_invalid_boxes(boxes, labels)
            sample["boxes"] = boxes
            if labels is not None:
                sample["labels"] = labels

        if "mask" in sample:
            sample["mask"] = cv2.resize(sample["mask"].astype(np.int64), (tw, th), interpolation=cv2.INTER_NEAREST)

        meta = sample.get("meta") or {}
        meta["scale"] = (sh, sw)
        meta["original_size"] = (h, w)
        sample["meta"] = meta
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_random_hflip")
class BoxSyncRandomHorizontalFlip(BaseTransform):
    """随机水平翻转 image + 同步翻转 boxes。"""

    name = "box_sync_random_hflip"

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        sample["image"] = cv2.flip(img, 1)
        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            sample["boxes"] = boxes
        if "mask" in sample:
            sample["mask"] = cv2.flip(sample["mask"], 1)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_random_crop")
class BoxSyncRandomCrop(BaseTransform):
    """随机裁剪 image，丢弃越界后为空的 box。"""

    name = "box_sync_random_crop"

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        h, w = img.shape[:2]
        th, tw = self.size
        if h < th or w < tw:
            return sample
        y = np.random.randint(0, h - th + 1)
        x = np.random.randint(0, w - tw + 1)
        sample["image"] = img[y:y + th, x:x + tw]
        if "mask" in sample:
            sample["mask"] = sample["mask"][y:y + th, x:x + tw]
        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th)
            labels = sample.get("labels")
            boxes, labels = _filter_invalid_boxes(boxes, labels)
            sample["boxes"] = boxes
            if labels is not None:
                sample["labels"] = labels
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_pad")
class BoxSyncPad(BaseTransform):
    """将 image pad 到目标尺寸（右下 pad），boxes 坐标不变。"""

    name = "box_sync_pad"

    def __init__(self, size: Tuple[int, int], pad_value: int = 0, mask_value: int = 0):
        self.size = size
        self.pad_value = pad_value
        self.mask_value = mask_value

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        h, w = img.shape[:2]
        th, tw = self.size
        if h >= th and w >= tw:
            return sample
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        sample["image"] = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.pad_value)
        if "mask" in sample:
            sample["mask"] = np.pad(sample["mask"], ((0, pad_h), (0, pad_w)), constant_values=self.mask_value)
        return sample