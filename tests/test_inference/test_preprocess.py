# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 推理预处理测试：规格提取、三类输入、反变换元信息、确定性与非法输入。
"""推理预处理测试：规格提取、三类输入、反变换元信息、确定性与非法输入。"""

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from dvisionix.data.transforms import ImageNormalize, ImageResize, TransformPipeline
from dvisionix.errors import InferenceError
from dvisionix.inference import PreprocessMeta, PreprocessPipeline, PreprocessSpec
from dvisionix.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD

TASKS = [
    ("classification", 64),
    ("multi_task_classification", 64),
    ("detection", 96),
    ("segmentation", 48),
    ("MaskFormer", 48),
]


def _image(height: int = 37, width: int = 53, seed: int = 0) -> np.ndarray:
    """构造确定性的 uint8 RGB(HWC) 测试图像。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class _FixedMetaPipeline:
    """把 meta 固定为与尺寸比不同的值的假流水线，用于验证"优先读 pipeline meta"。"""

    def __call__(self, sample):
        sample["image"] = torch.zeros(3, 10, 12, dtype=torch.float32)
        sample["meta"]["original_size"] = (20, 30)
        sample["meta"]["scale"] = (0.5, 0.4)
        return sample


class TestSpecExtraction:
    @pytest.mark.parametrize("task_type,image_size", TASKS)
    def test_from_task_extracts_image_size_and_normalization(self, task_type, image_size):
        prep = PreprocessPipeline.from_task(task_type, image_size)
        spec = prep.spec
        assert isinstance(spec, PreprocessSpec)
        assert spec.task_type == task_type
        assert spec.image_size == image_size
        assert spec.mean == IMAGENET_MEAN
        assert spec.std == IMAGENET_STD
        assert spec.scale == pytest.approx(1.0 / 255.0)

    @pytest.mark.parametrize(
        "task_type,image_size",
        [("classification", 32), ("detection", 48), ("segmentation", 40)],
    )
    def test_spec_from_pipeline_matches_task(self, task_type, image_size):
        from dvisionix.training.tasks import (
            ClassificationTask,
            DetectionTask,
            SegmentationTask,
        )

        task_cls = {
            "classification": ClassificationTask,
            "detection": DetectionTask,
            "segmentation": SegmentationTask,
        }[task_type]
        pipeline = task_cls.build_transforms(train=False, image_size=image_size)
        spec = PreprocessSpec.from_pipeline(pipeline, task_type=task_type)
        assert spec.task_type == task_type
        assert spec.image_size == image_size
        assert spec.mean == IMAGENET_MEAN
        assert spec.std == IMAGENET_STD

    def test_from_pipeline_uses_first_size_transform(self):
        pipeline = TransformPipeline([ImageResize((32, 48)), ImageResize((64, 64))])
        spec = PreprocessSpec.from_pipeline(pipeline)
        assert spec.image_size == 32

    def test_from_pipeline_custom_normalization(self):
        pipeline = TransformPipeline(
            [
                ImageResize((32, 32)),
                ImageNormalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), scale=1.0 / 127.5),
            ]
        )
        spec = PreprocessSpec.from_pipeline(pipeline, task_type="classification")
        assert spec.image_size == 32
        assert spec.mean == (0.5, 0.5, 0.5)
        assert spec.std == (0.25, 0.25, 0.25)
        assert spec.scale == pytest.approx(1.0 / 127.5)

    def test_from_pipeline_defaults_when_normalize_missing(self):
        pipeline = TransformPipeline([ImageResize((24, 24))])
        spec = PreprocessSpec.from_pipeline(pipeline)
        assert spec.image_size == 24
        assert spec.mean == IMAGENET_MEAN
        assert spec.std == IMAGENET_STD
        assert spec.scale == pytest.approx(1.0 / 255.0)

    def test_from_pipeline_without_size_raises(self):
        pipeline = TransformPipeline([ImageNormalize()])
        with pytest.raises(InferenceError, match="image_size"):
            PreprocessSpec.from_pipeline(pipeline)

    def test_from_task_unsupported_raises(self):
        with pytest.raises(InferenceError, match="不支持"):
            PreprocessPipeline.from_task("not_a_task", 32)

    def test_spec_dict_roundtrip(self):
        spec = PreprocessSpec(
            task_type="detection",
            image_size=320,
            mean=(0.5, 0.4, 0.3),
            std=(0.2, 0.3, 0.4),
            scale=1.0 / 127.5,
        )
        restored = PreprocessSpec.from_dict(spec.to_dict())
        assert restored == spec
        payload = spec.to_dict()
        assert payload["mean"] == [0.5, 0.4, 0.3]
        assert payload["std"] == [0.2, 0.3, 0.4]
        assert payload["image_size"] == 320

    def test_spec_from_dict_requires_image_size(self):
        with pytest.raises(InferenceError, match="image_size"):
            PreprocessSpec.from_dict({"task_type": "classification"})


class TestPreprocessCall:
    @pytest.mark.parametrize("task_type,image_size", TASKS)
    def test_output_tensor_and_meta(self, task_type, image_size):
        prep = PreprocessPipeline.from_task(task_type, image_size)
        array = _image(37, 53)
        tensor, meta = prep(array)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, image_size, image_size)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert isinstance(meta, PreprocessMeta)
        assert meta.original_size == (37, 53)
        assert meta.processed_size == (image_size, image_size)
        assert meta.image_path is None

    def test_ndarray_meta_scale_from_sizes(self):
        prep = PreprocessPipeline.from_task("classification", 64)
        _, meta = prep(_image(32, 48))
        assert meta.scale == pytest.approx((64 / 32, 64 / 48))

    def test_detection_meta_scale_written_by_pipeline(self):
        prep = PreprocessPipeline.from_task("detection", 96)
        _, meta = prep(_image(37, 53))
        # BoxSyncResize 写回的是 (th / h, tw / w)
        assert meta.original_size == (37, 53)
        assert meta.processed_size == (96, 96)
        assert meta.scale == pytest.approx((96 / 37, 96 / 53))

    def test_meta_prefers_values_written_by_pipeline(self):
        spec = PreprocessSpec(task_type="detection", image_size=10)
        prep = PreprocessPipeline(spec, pipeline=_FixedMetaPipeline())
        tensor, meta = prep(_image(20, 30))
        assert tensor.shape == (1, 3, 10, 12)
        assert meta.original_size == (20, 30)
        assert meta.scale == (0.5, 0.4)

    def test_path_input_bgr_to_rgb(self, tmp_path):
        prep = PreprocessPipeline.from_task("classification", 64)
        rgb = _image(40, 60, seed=3)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        path = tmp_path / "sample.png"
        assert cv2.imwrite(str(path), bgr)
        from_path, path_meta = prep(str(path))
        from_array, array_meta = prep(rgb)
        torch.testing.assert_close(from_path, from_array)
        assert path_meta.image_path == str(path)
        assert path_meta.original_size == (40, 60)
        assert array_meta.image_path is None

    def test_pathlib_path_input(self, tmp_path):
        import cv2

        prep = PreprocessPipeline.from_task("segmentation", 32)
        path = tmp_path / "sample.jpg"
        assert cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))
        tensor, meta = prep(path)
        assert tensor.shape == (1, 3, 32, 32)
        assert meta.image_path == str(path)

    def test_pil_input_matches_ndarray(self):
        prep = PreprocessPipeline.from_task("classification", 64)
        rgb = _image(40, 60, seed=5)
        from_pil, pil_meta = prep(Image.fromarray(rgb))
        from_array, _ = prep(rgb)
        torch.testing.assert_close(from_pil, from_array)
        assert pil_meta.image_path is None
        assert pil_meta.original_size == (40, 60)

    def test_pil_grayscale_converted_to_rgb(self):
        prep = PreprocessPipeline.from_task("classification", 32)
        gray = np.zeros((20, 30), dtype=np.uint8)
        tensor, meta = prep(Image.fromarray(gray))
        assert tensor.shape == (1, 3, 32, 32)
        assert meta.original_size == (20, 30)

    def test_grayscale_ndarray_expanded_to_three_channels(self):
        prep = PreprocessPipeline.from_task("classification", 32)
        tensor, _ = prep(np.zeros((20, 30), dtype=np.uint8))
        assert tensor.shape == (1, 3, 32, 32)

    @pytest.mark.parametrize("task_type", ["classification", "detection", "segmentation"])
    def test_deterministic_and_input_not_modified(self, task_type):
        prep = PreprocessPipeline.from_task(task_type, 48)
        array = _image(37, 53, seed=7)
        snapshot = array.copy()
        first, first_meta = prep(array)
        second, second_meta = prep(array)
        assert torch.equal(first, second)
        assert first_meta.to_dict() == second_meta.to_dict()
        np.testing.assert_array_equal(array, snapshot)

    def test_pipeline_constructed_from_spec(self):
        spec = PreprocessSpec(task_type="detection", image_size=40)
        prep = PreprocessPipeline(spec)
        tensor, meta = prep(_image(20, 30))
        assert tensor.shape == (1, 3, 40, 40)
        assert meta.processed_size == (40, 40)
        assert prep.spec == spec

    def test_meta_to_dict_json_friendly(self):
        prep = PreprocessPipeline.from_task("detection", 40)
        _, meta = prep(_image(20, 30))
        payload = meta.to_dict()
        assert payload["original_size"] == [20, 30]
        assert payload["processed_size"] == [40, 40]
        assert payload["scale"][0] == 2.0
        assert payload["scale"][1] == pytest.approx(40 / 30)
        assert payload["image_path"] is None
        assert all(isinstance(value, list) for value in payload.values() if value is not None)

    @pytest.mark.parametrize("bad_input", [None, 123, [1, 2, 3], b"bytes", object()])
    def test_unsupported_input_type_raises(self, bad_input):
        prep = PreprocessPipeline.from_task("classification", 32)
        with pytest.raises(InferenceError, match="不支持"):
            prep(bad_input)

    def test_missing_path_raises(self, tmp_path):
        prep = PreprocessPipeline.from_task("classification", 32)
        with pytest.raises(InferenceError, match="无法读取图像"):
            prep(tmp_path / "missing.jpg")

    def test_float_ndarray_raises(self):
        prep = PreprocessPipeline.from_task("classification", 32)
        with pytest.raises(InferenceError, match="dtype"):
            prep(np.zeros((20, 30, 3), dtype=np.float32))

    def test_invalid_channel_count_raises(self):
        prep = PreprocessPipeline.from_task("classification", 32)
        with pytest.raises(InferenceError, match="通道数"):
            prep(np.zeros((20, 30, 5), dtype=np.uint8))

    def test_invalid_ndim_raises(self):
        prep = PreprocessPipeline.from_task("classification", 32)
        with pytest.raises(InferenceError, match="形状"):
            prep(np.zeros((2, 20, 30, 3), dtype=np.uint8))

    def test_non_positive_image_size_spec_raises(self):
        with pytest.raises(InferenceError, match="image_size"):
            PreprocessSpec(task_type="classification", image_size=0)
