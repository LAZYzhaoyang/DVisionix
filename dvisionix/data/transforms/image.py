# -*- coding: utf-8 -*-
"""原子图像变换（任务无关，只动 ``image`` 字段）。

约定：
- 输入 ``image`` 字段为 numpy uint8 (H, W, C)，RGB 顺序。
- 输出保持 numpy 数组，最后一步 ToTensor 才转 torch.Tensor。
- 几何相关且需要同步处理 box/mask 的变换放在 ``geometric.py``。
"""

from typing import Tuple

import cv2
import numpy as np

from ..sample import Sample
from .base import BaseTransform
from ...registry import TRANSFORMS


@TRANSFORMS.register()
@TRANSFORMS.register(name="image_resize")
class ImageResize(BaseTransform):
    """仅调整 image 大小（不同步处理 box/mask）。"""

    name = "image_resize"

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        if isinstance(img, np.ndarray):
            sample["image"] = cv2.resize(img, (self.size[1], self.size[0]))
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="random_hflip")
class RandomHorizontalFlip(BaseTransform):
    """随机水平翻转（仅 image）。"""

    name = "random_hflip"

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() < self.p:
            sample["image"] = cv2.flip(sample["image"], 1)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="random_vflip")
class RandomVerticalFlip(BaseTransform):
    """随机垂直翻转（仅 image）。"""

    name = "random_vflip"

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() < self.p:
            sample["image"] = cv2.flip(sample["image"], 0)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="random_crop")
class RandomCrop(BaseTransform):
    """随机裁剪（仅 image）。"""

    name = "random_crop"

    def __init__(self, size: Tuple[int, int] = (224, 224)):
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
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="center_crop")
class CenterCrop(BaseTransform):
    name = "center_crop"

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        h, w = img.shape[:2]
        th, tw = self.size
        y = (h - th) // 2
        x = (w - tw) // 2
        sample["image"] = img[y:y + th, x:x + tw]
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="color_jitter")
class ColorJitter(BaseTransform):
    """亮度/对比度/饱和度随机扰动（仅 image）。"""

    name = "color_jitter"

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"].astype(np.float32)
        if self.brightness > 0:
            a = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            img = np.clip(img * a, 0, 255)
        if self.contrast > 0:
            a = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            gray = float(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).mean())
            img = np.clip(a * img + (1 - a) * gray, 0, 255)
        if self.saturation > 0:
            a = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = np.stack([gray] * 3, axis=-1).astype(np.float32)
            img = np.clip(a * img + (1 - a) * gray, 0, 255)
        sample["image"] = img.astype(np.uint8)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="normalize")
class ImageNormalize(BaseTransform):
    """像素归一化（uint8 -> float32 / scale -> (x - mean) / std）。"""

    name = "normalize"
    provides_normalization = True

    def __init__(self, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 scale: float = 1.0 / 255.0):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.scale = float(scale)

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"].astype(np.float32) * self.scale
        sample["image"] = (img - self.mean) / self.std
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="to_tensor")
class ToTensor(BaseTransform):
    """numpy (H, W, C) -> torch.Tensor (C, H, W) float32。"""

    name = "to_tensor"

    def __init__(self, keys: Tuple[str, ...] = ("image",)):
        self.keys = keys

    def __call__(self, sample: Sample) -> Sample:
        import torch
        for k in self.keys:
            if k not in sample:
                continue
            v = sample[k]
            if isinstance(v, np.ndarray):
                if v.ndim == 3:
                    t = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1))).float()
                else:
                    t = torch.from_numpy(np.ascontiguousarray(v))
                    if t.dtype != torch.float32 and k in ("image", "mask"):
                        t = t.float()
                sample[k] = t
        return sample