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


class TestCropContracts:
    """裁剪契约（CodePlan 7.4 步骤 2-3 / 7.1.2 D9）。

    v1.0.0 的裁剪在输入过小时没有任何提示：``RandomCrop`` 静默返回原图、
    ``CenterCrop`` 经负索引切片静默返回更小的图，二者都会产出尺寸错误的张量。
    """

    def test_random_crop_raises_on_small_input_by_default(self):
        from dvisionix.data.transforms import RandomCrop

        with pytest.raises(ValueError, match="小于目标裁剪尺寸"):
            RandomCrop((64, 64))({"image": _img(32, 32)})

    def test_center_crop_raises_on_small_input_by_default(self):
        from dvisionix.data.transforms import CenterCrop

        with pytest.raises(ValueError, match="小于目标裁剪尺寸"):
            CenterCrop((64, 64))({"image": _img(32, 64)})

    def test_on_small_pad_pads_to_target_size(self):
        from dvisionix.data.transforms import CenterCrop, RandomCrop

        for op in (RandomCrop((64, 64), on_small="pad"), CenterCrop((64, 64), on_small="pad")):
            out = op({"image": _img(32, 32)})
            assert out["image"].shape == (64, 64, 3)

    def test_on_small_resize_resizes_to_target_size(self):
        from dvisionix.data.transforms import CenterCrop, RandomCrop

        for op in (
            RandomCrop((64, 64), on_small="resize"),
            CenterCrop((64, 64), on_small="resize"),
        ):
            out = op({"image": _img(32, 48)})
            assert out["image"].shape == (64, 64, 3)

    def test_invalid_on_small_value_raises(self):
        from dvisionix.data.transforms import RandomCrop

        with pytest.raises(ValueError, match="on_small"):
            RandomCrop((64, 64), on_small="whatever")({"image": _img(32, 32)})

    def test_large_enough_input_is_cropped_exactly(self):
        from dvisionix.data.transforms import CenterCrop

        out = CenterCrop((48, 48))({"image": _img(64, 64)})
        assert out["image"].shape == (48, 48, 3)


class TestGeometricCropContract:
    """几何同步裁剪：必须真产生随机偏移，且 image/boxes/mask 保持一致。"""

    def _sample(self, size=70, box=(30.0, 30.0, 40.0, 40.0)):
        from dvisionix.data.transforms import BoxSyncRandomCrop  # noqa: F401

        return {
            "image": _img(size, size),
            "mask": np.zeros((size, size), dtype=np.int64),
            "boxes": np.array([box], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }

    def test_geometric_crop_raises_on_small_input_by_default(self):
        from dvisionix.data.transforms import BoxSyncRandomCrop

        with pytest.raises(ValueError, match="小于目标裁剪尺寸"):
            BoxSyncRandomCrop((64, 64))(self._sample(size=32))

    def test_geometric_crop_has_nonzero_and_varying_offsets(self):
        """裁剪必须产生非零随机偏移（旧实现下偏移恒为 0，增强完全失效）。"""
        from dvisionix.data.transforms import BoxSyncRandomCrop

        op = BoxSyncRandomCrop((64, 64))
        results = set()
        for seed in range(16):
            np.random.seed(seed)
            out = op(self._sample(size=70))
            results.add(tuple(out["boxes"][0].tolist()))
        assert len(results) > 1, f"裁剪后 boxes 始终相同，说明偏移恒为 0：{results}"
        # 裁剪偏移恒 >= 0，因此框只会相对左上角内移；出现 x1 < 30 即证明偏移非零
        assert any(r[0] < 30.0 for r in results), results
        assert all(r[0] <= 30.0 and r[1] <= 30.0 for r in results), results

    def test_geometric_crop_keeps_image_boxes_mask_in_sync(self):
        from dvisionix.data.transforms import BoxSyncRandomCrop

        np.random.seed(3)
        out = BoxSyncRandomCrop((64, 64))(self._sample(size=70))
        assert out["image"].shape == (64, 64, 3)
        assert out["mask"].shape == (64, 64)
        assert len(out["boxes"]) == len(out["labels"])
        boxes = out["boxes"]
        assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 64).all()
        assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= 64).all()

    def test_detection_pipeline_training_actually_crops(self):
        """检测训练预置管线必须"先放大再裁剪"（D9 的端到端回归）。"""
        from dvisionix.data.transforms import DetectionTransforms

        results = set()
        for seed in range(8):
            np.random.seed(seed)
            out = DetectionTransforms(train=True, image_size=64)(self._sample(size=64))
            assert out["image"].shape == (3, 64, 64)
            results.add(tuple(out["boxes"][0].tolist()))
        assert (
            len(results) > 1
        ), f"检测训练管线输出恒定，随机裁剪未生效（resize 与 crop 尺寸相同）：{results}"

    def test_detection_pipeline_eval_is_deterministic(self):
        from dvisionix.data.transforms import DetectionTransforms

        tf = DetectionTransforms(train=False, image_size=64)
        first = tf(self._sample(size=64))["boxes"][0].tolist()
        second = tf(self._sample(size=64))["boxes"][0].tolist()
        assert first == second


class TestBoxesLabelsConsistency:
    """几何变换入口必须校验 boxes 与 labels 数量一致。"""

    @pytest.mark.parametrize(
        "op_name",
        ["BoxSyncResize", "BoxSyncRandomHorizontalFlip", "BoxSyncRandomCrop"],
    )
    def test_mismatched_boxes_and_labels_raise(self, op_name):
        import dvisionix.data.transforms as T

        op_cls = getattr(T, op_name)
        op = op_cls((32, 32)) if op_name != "BoxSyncRandomHorizontalFlip" else op_cls(p=1.0)
        sample = {
            "image": _img(64, 64),
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        with pytest.raises(ValueError, match="数量不一致"):
            op(sample)

    def test_matching_boxes_and_labels_pass(self):
        from dvisionix.data.transforms import BoxSyncResize

        sample = {
            "image": _img(64, 64),
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        out = BoxSyncResize((32, 32))(sample)
        assert len(out["boxes"]) == len(out["labels"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
