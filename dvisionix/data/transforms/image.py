# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 原子图像变换（任务无关，只动 ``image`` 字段）。
"""原子图像变换（任务无关，只动 ``image`` 字段）。

约定：
- 输入 ``image`` 字段为 numpy uint8 (H, W, C)，RGB 顺序。
- 输出保持 numpy 数组，最后一步 ToTensor 才转 torch.Tensor。
- 几何相关且需要同步处理 box/mask 的变换放在 ``geometric.py``。
"""

from typing import Tuple

import cv2
import numpy as np

from ...registry import TRANSFORMS
from ..sample import Sample
from .base import BaseTransform


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


def _ensure_crop_size(img: np.ndarray, size: Tuple[int, int], on_small: str, op: str) -> np.ndarray:
    """确保图像不小于裁剪尺寸，按 ``on_small`` 策略处理过小的输入。

    v1.0.0 的行为是**静默返回原图**（``RandomCrop``）或**静默返回更小的裁剪**
    （``CenterCrop``：负索引切片），两者都会产出尺寸错误的张量而没有提示
    （CodePlan 7.1.2 D9）。现在把策略显式化。

    Args:
        img: (H, W, C) 输入图像。
        size: 目标裁剪尺寸 ``(th, tw)``。
        on_small: 输入小于目标时的行为。``error``（默认）直接报错；``pad`` 右下补零；
            ``resize`` 直接缩放到目标尺寸。
        op: 调用方名称，用于错误信息定位。

    Returns:
        不小于 ``size`` 的图像（``error`` 策略下尺寸不足时抛错）。
    """
    import cv2

    h, w = img.shape[:2]
    th, tw = size
    if h >= th and w >= tw:
        return img
    if on_small == "error":
        raise ValueError(
            f"{op}: 输入尺寸 {h}x{w} 小于目标裁剪尺寸 {th}x{tw}。"
            f"请先用 resize 放大到至少 {th}x{tw}，"
            f"或显式设置 on_small='pad' / on_small='resize'。"
        )
    if on_small == "pad":
        return cv2.copyMakeBorder(
            img, 0, max(0, th - h), 0, max(0, tw - w), cv2.BORDER_CONSTANT, value=0
        )
    if on_small == "resize":
        return cv2.resize(img, (tw, th))
    raise ValueError(f"{op}: on_small 必须是 'error' / 'pad' / 'resize'，当前 {on_small!r}")


@TRANSFORMS.register()
@TRANSFORMS.register(name="random_crop")
class RandomCrop(BaseTransform):
    """随机裁剪（仅 image）。

    Args:
        size: 目标 ``(H, W)``。
        on_small: 输入小于目标尺寸时的行为，见 ``_ensure_crop_size``。
            v1.0.0 为静默返回原图（输出尺寸错误），现默认 ``error``。
    """

    name = "random_crop"

    def __init__(self, size: Tuple[int, int] = (224, 224), on_small: str = "error"):
        self.size = size
        self.on_small = on_small

    def __call__(self, sample: Sample) -> Sample:
        img = _ensure_crop_size(sample["image"], self.size, self.on_small, "RandomCrop")
        h, w = img.shape[:2]
        th, tw = self.size
        y = np.random.randint(0, h - th + 1)
        x = np.random.randint(0, w - tw + 1)
        sample["image"] = img[y : y + th, x : x + tw]
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="center_crop")
class CenterCrop(BaseTransform):
    """中心裁剪（仅 image）。

    Args:
        size: 目标 ``(H, W)``。
        on_small: 输入小于目标尺寸时的行为，见 ``_ensure_crop_size``。
            v1.0.0 无任何检查，会经负索引切片静默返回更小的图，现默认 ``error``。
    """

    name = "center_crop"

    def __init__(self, size: Tuple[int, int] = (224, 224), on_small: str = "error"):
        self.size = size
        self.on_small = on_small

    def __call__(self, sample: Sample) -> Sample:
        img = _ensure_crop_size(sample["image"], self.size, self.on_small, "CenterCrop")
        h, w = img.shape[:2]
        th, tw = self.size
        y = (h - th) // 2
        x = (w - tw) // 2
        sample["image"] = img[y : y + th, x : x + tw]
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

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: float = 1.0 / 255.0,
    ):
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
            if not isinstance(v, np.ndarray):
                continue
            if k == "mask":
                # 分割标签必须满足 CrossEntropyLoss 的 **long** 契约。
                # v1.0.0 只在 2 维分支做 dtype 处理，3 维 mask 会走上面的 float 分支
                # 而静默变成 float32（CodePlan 7.1.2 D10）；这里对 mask 单独短路，
                # 与 ndim 无关地保持 long，也不做通道前置（长整型标签不需要 C,H,W 布局）。
                sample[k] = torch.from_numpy(np.ascontiguousarray(v)).long()
                continue
            if v.ndim == 3:
                sample[k] = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1))).float()
            else:
                t = torch.from_numpy(np.ascontiguousarray(v))
                sample[k] = t.float() if k == "image" else t
        return sample
