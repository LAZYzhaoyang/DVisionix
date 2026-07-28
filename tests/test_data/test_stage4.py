# -*- coding: utf-8 -*-
"""阶段 4 数据管线测试：归一化职责唯一化 / DetectionResize 越界裁剪 / Albumentations bbox。"""

import os
import sys
import tempfile
import warnings

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.data import (
    CustomDataset,
    ClassificationTransforms,
    DetectionTransforms,
    SegmentationTransforms,
)
from dvisionix.data.transforms.detection import DetectionResize


def _make_image_file(tmp: str, name: str = "img.png") -> str:
    path = os.path.join(tmp, name)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


class TestNormalizationOwnership:
    """归一化只应发生一次。"""

    def test_transforms_carry_normalization_flag(self):
        cls_tf = ClassificationTransforms(train=False, image_size=32)
        det_tf = DetectionTransforms(train=False, image_size=32, max_size=32)
        seg_tf = SegmentationTransforms(train=False, image_size=32)
        assert cls_tf.provides_normalization is True
        assert det_tf.provides_normalization is True
        assert seg_tf.provides_normalization is True

    def test_dataset_skips_internal_normalize_when_transforms_do_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = _make_image_file(tmp)
            transforms = ClassificationTransforms(train=False, image_size=32)
            ds = CustomDataset(
                task_type="classification",
                samples=[{"image_path": img_path, "label": 0}],
                num_classes=2,
                transforms=transforms,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                sample = ds[0]  # 不应触发 DeprecationWarning
            assert sample["image"].shape == (3, 32, 32)
            # 已归一化：均值/方差应接近 0/1 附近而非 [0,255]
            assert sample["image"].abs().max() < 50

    def test_dataset_emits_deprecation_when_transforms_missing_normalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = _make_image_file(tmp)
            ds = CustomDataset(
                task_type="classification",
                samples=[{"image_path": img_path, "label": 0}],
                num_classes=2,
                transforms=None,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _ = ds[0]
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)


class TestDetectionResizeClipping:
    """越界坐标应被裁剪回图像范围内。"""

    def test_boxes_clipped_to_target_bounds(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 故意构造超出原图边界的框，验证裁剪逻辑
        boxes = np.array([
            [-10.0, -10.0, 120.0, 120.0],
            [50.0, 50.0, 80.0, 80.0],
        ], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        data = {"image": image, "boxes": boxes.copy(), "labels": labels}

        resized = DetectionResize(size=(50, 50))(data)
        out = resized["boxes"]
        assert (out[:, [0, 2]] >= 0).all()
        assert (out[:, [1, 3]] >= 0).all()
        assert (out[:, [0, 2]] <= 50).all()
        assert (out[:, [1, 3]] <= 50).all()

    def test_degenerate_boxes_filtered(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 一个完全在图外的框（缩放后仍应被过滤）
        boxes = np.array([
            [150.0, 150.0, 200.0, 200.0],
            [10.0, 10.0, 30.0, 30.0],
        ], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        data = {"image": image, "boxes": boxes.copy(), "labels": labels}

        resized = DetectionResize(size=(50, 50))(data)
        # 第一个框在裁剪后应退化为零面积并被丢弃
        assert len(resized["boxes"]) == 1
        assert len(resized["labels"]) == 1