# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: BaseDataset mask 路径加载测试（Sample 契约：mask 可为文件路径）。
"""BaseDataset mask 路径加载测试（Sample 契约：mask 可为文件路径）。"""

import os
import tempfile

import cv2
import numpy as np
import torch

from dvisionix.data import CustomDataset
from dvisionix.data.transforms import SegmentationTransforms


def test_mask_path_loading():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "img.png")
        mask_path = os.path.join(tmp, "mask.png")
        cv2.imwrite(img_path, np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        mask = np.random.randint(0, 3, (32, 32), dtype=np.uint8)
        cv2.imwrite(mask_path, mask)

        ds = CustomDataset(
            samples=[{"image": img_path, "mask": mask_path}],
            task_type="segmentation",
            transforms=SegmentationTransforms(train=False, image_size=16),
        )
        sample = ds[0]
        assert "image" in sample and "mask" in sample
        # SegmentationTransforms 末尾 MaskToTensor 已转 long tensor
        assert sample["mask"].dtype == torch.long
        assert sample["mask"].shape == (16, 16)
        assert sample["image"].shape == (3, 16, 16)
