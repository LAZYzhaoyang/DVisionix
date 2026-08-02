# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 变换测试：原子 transform + 几何同步 + pipeline + 注册。
"""变换测试：原子 transform + 几何同步 + pipeline + 注册。"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.data.transforms import (
    AlbumentationsWrapper,
    BoxSyncRandomHorizontalFlip,
    BoxSyncResize,
    ImageNormalize,
    ImageResize,
    ToTensor,
    TransformPipeline,
    build_pipeline,
    build_transform,
)
from dvisionix.registry import TRANSFORMS


def _img(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestImageAtoms:
    def test_image_resize(self):
        t = ImageResize((32, 32))
        out = t({"image": _img(64, 64)})
        assert out["image"].shape == (32, 32, 3)

    def test_to_tensor_layout(self):
        t = ToTensor(keys=("image",))
        out = t({"image": _img(16, 16)})
        assert isinstance(out["image"], torch.Tensor) and out["image"].shape == (3, 16, 16)

    def test_normalize_provides_flag(self):
        assert ImageNormalize().provides_normalization is True


class TestGeometricSync:
    def test_box_sync_resize_keeps_within_bounds(self):
        img = _img(100, 100)
        boxes = np.array([[150, 150, 200, 200], [10, 10, 30, 30]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        out = BoxSyncResize(size=(50, 50))({"image": img, "boxes": boxes, "labels": labels})
        assert (out["boxes"][:, [0, 2]] >= 0).all() and (out["boxes"][:, [0, 2]] <= 50).all()
        assert (out["boxes"][:, [1, 3]] >= 0).all() and (out["boxes"][:, [1, 3]] <= 50).all()
        # 第一个框因 w<=0 退化被丢弃
        assert len(out["boxes"]) == 1

    def test_box_sync_resize_with_mask(self):
        img = _img(100, 100)
        mask = np.zeros((100, 100), dtype=np.int64)
        out = BoxSyncResize(size=(40, 50))({"image": img, "mask": mask})
        assert out["mask"].shape == (40, 50)

    def test_hflip_flips_boxes_and_mask(self):
        img = _img(20, 30)
        boxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
        mask = np.arange(600, dtype=np.int64).reshape(20, 30)
        np.random.seed(0)
        out = BoxSyncRandomHorizontalFlip(p=1.0)(
            {"image": img, "boxes": boxes.copy(), "mask": mask}
        )
        np.testing.assert_array_equal(out["boxes"][0, [0, 2]], [30 - 15, 30 - 5])
        np.testing.assert_array_equal(out["mask"], np.fliplr(mask))


class TestPipeline:
    def test_pipeline_runs_in_order(self):
        pipe = TransformPipeline([ImageResize((32, 32)), ToTensor(keys=("image",))])
        out = pipe({"image": _img(64, 64)})
        assert out["image"].shape == (3, 32, 32)

    def test_pipeline_propagates_normalization_flag(self):
        pipe = TransformPipeline([ImageResize((16, 16)), ImageNormalize()])
        assert pipe.provides_normalization is True

    def test_append_chain(self):
        pipe = TransformPipeline([ImageResize((8, 8))])
        pipe.append(ToTensor(keys=("image",)))
        assert len(pipe) == 2

    def test_build_from_mixed_specs(self):
        pipe = build_pipeline(
            [
                {"type": "image_resize", "size": [16, 16]},
                "to_tensor",
            ]
        )
        out = pipe({"image": _img(32, 32)})
        assert out["image"].shape == (3, 16, 16)


class TestRegistry:
    def test_atomic_registered(self):
        for name in [
            "image_resize",
            "random_hflip",
            "to_tensor",
            "normalize",
            "box_sync_resize",
            "box_sync_random_hflip",
            "label_to_tensor",
            "boxes_to_tensor",
            "mask_to_tensor",
            "albumentations",
        ]:
            assert name in TRANSFORMS

    def test_build_transform_from_str(self):
        t = build_transform("image_resize")
        assert isinstance(t, ImageResize)


class TestThirdParty:
    def test_albumentations_classification(self):
        pytest.importorskip("albumentations")
        import albumentations as A

        albu = A.Compose([A.Resize(20, 20), A.HorizontalFlip(p=0.0)])
        t = AlbumentationsWrapper(albu, is_detection=False, is_segmentation=False)
        out = t({"image": _img(64, 64)})
        assert out["image"].shape == (20, 20, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
