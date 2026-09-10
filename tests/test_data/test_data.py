# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 数据模块测试：Sample 契约 + BaseDataset + CustomDataset + 主流公开数据集。
"""数据模块测试：Sample 契约 + BaseDataset + CustomDataset + 主流公开数据集。"""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.data import (
    BaseDataset,
    CustomDataset,
    Sample,
    build_dataset,
    segmentation_collate,
)
from dvisionix.registry import DATASETS


def _write_img(path: str, size=(32, 32)) -> str:
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


class TestSampleContract:
    def test_sample_attribute_access(self):
        s = Sample(image=np.zeros((2, 2, 3), dtype=np.uint8), label=0)
        assert s.image.shape == (2, 2, 3)
        assert s["label"] == 0


class TestBaseDataset:
    def test_loads_image_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_img(os.path.join(tmp, "a.png"))
            ds = BaseDataset(samples=[{"image": p, "label": 0}])
            out = ds[0]
            assert isinstance(out["image"], np.ndarray) and out["image"].shape == (32, 32, 3)
            assert out["label"] == 0

    def test_passes_through_when_already_ndarray(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        ds = BaseDataset(samples=[{"image": img, "label": 1}])
        out = ds[0]
        assert out["image"] is img

    def test_task_type_default(self):
        ds = BaseDataset(samples=[{"image": "x", "label": 0}])
        assert ds.task_type == ""


class TestCustomDataset:
    def test_classification_minimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                {"image": _write_img(os.path.join(tmp, f"{i}.png")), "label": i % 2}
                for i in range(3)
            ]
            ds = CustomDataset(samples=samples, task_type="classification")
            assert len(ds) == 3
            assert "image" in ds[0]

    def test_detection_uses_detection_collate(self):
        with tempfile.TemporaryDirectory() as tmp:
            from dvisionix.data.transforms import ToTensor

            samples = [
                {
                    "image": _write_img(os.path.join(tmp, "0.png")),
                    "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
                    "labels": np.array([0], dtype=np.int64),
                }
            ]
            ds = CustomDataset(
                samples=samples, task_type="detection", transforms=ToTensor(keys=("image",))
            )
            assert ds.collate_fn is not None
            batch = [ds[0], ds[0]]
            out = ds.collate_fn(batch)
            assert out["image"].shape == (2, 3, 32, 32)
            assert isinstance(out["boxes"], list) and len(out["boxes"]) == 2

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            CustomDataset(samples=[], task_type="classification")


class TestRegistration:
    def test_mainstream_datasets_registered(self):
        for name in [
            "cifar10",
            "cifar100",
            "imagenet",
            "imagefolder",
            "coco",
            "coco_detection",
            "voc_detection",
            "voc_segmentation",
            "cityscapes",
            "ade20k",
            "custom",
            "base_dataset",
        ]:
            assert name in DATASETS, name

    def test_build_dataset_via_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            # imagefolder：建立 root/cls/img.jpg 结构
            for cls in ("a", "b"):
                cls_dir = os.path.join(tmp, cls)
                os.makedirs(cls_dir, exist_ok=True)
                _write_img(os.path.join(cls_dir, "x.png"))
            ds = build_dataset({"type": "imagefolder", "root": tmp})
            assert len(ds) == 2
            assert ds.classes == ["a", "b"]


class TestCollate:
    def test_segmentation_collate_stacks(self):
        s1 = {"image": torch.zeros(3, 8, 8), "mask": torch.zeros(8, 8, dtype=torch.long)}
        s2 = {"image": torch.ones(3, 8, 8), "mask": torch.ones(8, 8, dtype=torch.long)}
        out = segmentation_collate([s1, s2])
        assert out["image"].shape == (2, 3, 8, 8)
        assert out["mask"].shape == (2, 8, 8)


class TestCustomDatasetCollate:
    """CustomDataset 的 collate_fn 契约（CodePlan 7.3 步骤 1-5）。

    v1.0.0 用 ``staticmethod(detection_collate)`` 写实例属性。实例属性不经过描述符协议，
    该值会以 staticmethod 对象原样存下来；它只在 Python 3.10+ 才可调用，
    而项目最低版本恰好是 3.10，所以行为「碰巧正确」但依赖解释器版本。
    现在改为直接存函数对象，并在构造期校验可调用性。
    """

    @staticmethod
    def _detection_samples(n: int = 3):
        return [
            {
                "image": torch.randn(3, 8, 8),
                "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                "labels": torch.tensor([0]),
            }
            for _ in range(n)
        ]

    @staticmethod
    def _segmentation_samples(n: int = 3):
        return [
            {"image": torch.randn(3, 8, 8), "mask": torch.zeros(8, 8, dtype=torch.long)}
            for _ in range(n)
        ]

    def test_collate_fn_is_callable_for_every_task(self):
        cls_ds = CustomDataset(
            samples=[{"image": torch.randn(3, 8, 8), "label": 0}], task_type="classification"
        )
        # 分类沿用 PyTorch 默认 collate
        assert cls_ds.collate_fn is None

        det_ds = CustomDataset(samples=self._detection_samples(), task_type="detection")
        seg_ds = CustomDataset(samples=self._segmentation_samples(), task_type="segmentation")
        for ds in (det_ds, seg_ds):
            assert callable(ds.collate_fn), f"collate_fn 不可调用: {type(ds.collate_fn)}"

    def test_user_collate_fn_has_highest_priority(self):
        def sentinel(batch):
            return {"custom": True}

        ds = CustomDataset(
            samples=self._detection_samples(), task_type="detection", collate_fn=sentinel
        )
        assert ds.collate_fn is sentinel

    def test_detection_dataloader_yields_one_batch(self):
        ds = CustomDataset(samples=self._detection_samples(4), task_type="detection")
        loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 3, 8, 8)
        assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
        assert isinstance(batch["labels"], list) and len(batch["labels"]) == 2

    def test_segmentation_dataloader_yields_long_mask(self):
        ds = CustomDataset(samples=self._segmentation_samples(4), task_type="segmentation")
        loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 3, 8, 8)
        assert batch["mask"].shape == (2, 8, 8)
        assert batch["mask"].dtype == torch.long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
