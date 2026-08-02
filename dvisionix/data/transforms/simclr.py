# -*- coding: utf-8 -*-
"""SimCLR 双视角增强（随机裁剪/翻转/色彩抖动 + 归一化）。"""

from typing import Tuple

import numpy as np
import torch

from ...registry import TRANSFORMS
from ..sample import Sample
from .base import BaseTransform


@TRANSFORMS.register()
@TRANSFORMS.register(name="simclr_transforms")
class SimCLRTransforms(BaseTransform):
    """SimCLR 风格双视角增强：对同一图像生成两个增强视图（image1 / image2）。

    输出 Sample 含 ``image1`` / ``image2``（(C,H,W) float 张量，已归一化）；
    供 ``SimCLRTask`` 与 ``InfoNCELoss`` 端到端对比学习使用。
    """

    provides_normalization = True

    def __init__(
        self,
        image_size: int = 64,
        train: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        jitter: float = 0.4,
    ):
        self.image_size = int(image_size)
        self.train = bool(train)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.jitter = float(jitter)

    def __call__(self, sample: Sample) -> Sample:
        img = np.asarray(sample["image"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32)
        out = dict(sample)
        out["image1"] = self._augment(img)
        out["image2"] = self._augment(img)
        return out

    def _augment(self, img: np.ndarray) -> torch.Tensor:
        if self.train:
            img = self._random_resized_crop(img, self.image_size)
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            img = self._color_jitter(img)
        else:
            img = self._resize(img, self.image_size)
        img = img.clip(0, 255)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.from_numpy(self.mean).view(3, 1, 1)
        std = torch.from_numpy(self.std).view(3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _resize(img: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((size, size)))

    def _random_resized_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        from PIL import Image

        h, w = img.shape[:2]
        scale = np.random.uniform(0.4, 1.0)
        crop_h = max(int(h * np.sqrt(scale)), size)
        crop_w = max(int(w * np.sqrt(scale)), size)
        y0 = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        x0 = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        crop = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((size, size)))

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """亮度 / 对比度 / 饱和度抖动（HSV 空间饱和度）。"""
        j = self.jitter
        if np.random.rand() < 0.8:
            b = 1.0 + np.random.uniform(-j, j)
            img = img * b
        if np.random.rand() < 0.8:
            c = 1.0 + np.random.uniform(-j, j)
            gray = img.mean(axis=2, keepdims=True)
            img = gray + c * (img - gray)
        if np.random.rand() < 0.8:
            from PIL import Image

            hsv = np.asarray(
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert("HSV")
            ).astype(np.float32)
            s = 1.0 + np.random.uniform(-j, j)
            hsv[..., 1] = hsv[..., 1] * s
            img = np.asarray(
                Image.fromarray(hsv.clip(0, 255).astype(np.uint8)).convert("RGB")
            ).astype(np.float32)
        return img


__all__ = ["SimCLRTransforms"]
