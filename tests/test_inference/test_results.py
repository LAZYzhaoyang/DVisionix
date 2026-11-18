# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 推理结果测试：JSON 往返、文件读写与三种任务的可视化。
"""推理结果测试：JSON 往返、文件读写与三种任务的可视化。"""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from dvisionix.errors import InferenceError
from dvisionix.inference import Results


def _classification_results() -> Results:
    """构造一个分类结果（probs 和为 1，top-k 降序）。"""
    return Results(
        task_type="classification",
        orig_size=(48, 64),
        probs=np.array([0.1, 0.7, 0.2], dtype=np.float32),
        topk=[(1, 0.7), (2, 0.2), (0, 0.1)],
        meta={"num_classes": 3},
    )


def _detection_results() -> Results:
    """构造一个检测结果（框为原图坐标，含两个类别）。"""
    return Results(
        task_type="detection",
        orig_size=(64, 80),
        image_path="demo/det.jpg",
        boxes=np.array([[5, 6, 30, 40], [40, 20, 72, 70]], dtype=np.float32),
        scores=np.array([0.9, 0.55], dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int64),
        meta={"耗时": 0.012, "names": ["cat", "dog"]},
    )


def _segmentation_results() -> Results:
    """构造一个分割结果（mask 为原图尺寸的 int64 类别图）。"""
    mask = np.zeros((40, 50), dtype=np.int64)
    mask[:20] = 1
    mask[:, :25] += 2
    return Results(task_type="segmentation", orig_size=(40, 50), mask=mask)


class TestSerialization:
    def test_to_dict_numpy_to_list(self):
        results = _detection_results()
        payload = results.to_dict()
        assert payload["task_type"] == "detection"
        assert payload["orig_size"] == [64, 80]
        assert payload["image_path"] == "demo/det.jpg"
        assert isinstance(payload["boxes"], list)
        assert isinstance(payload["scores"], list)
        assert isinstance(payload["labels"], list)
        assert payload["boxes"] == results.boxes.tolist()
        assert payload["labels"] == results.labels.tolist()
        assert "image" not in payload

    def test_mask_to_dict_is_plain_ints(self):
        results = _segmentation_results()
        payload = results.to_dict()
        assert isinstance(payload["mask"], list)
        assert payload["mask"] == results.mask.tolist()
        assert payload["probs"] is None and payload["boxes"] is None

    def test_json_text_is_indented_and_utf8(self):
        results = _detection_results()
        text = results.to_json()
        assert "\n  " in text
        assert "耗时" in text  # ensure_ascii=False
        assert json.loads(text) == results.to_dict()

    def test_to_json_writes_file_and_returns_same_text(self, tmp_path):
        results = _classification_results()
        path = tmp_path / "cls.json"
        text = results.to_json(str(path))
        assert path.read_text(encoding="utf-8") == text
        assert json.loads(text)["topk"] == [[1, 0.7], [2, 0.2], [0, 0.1]]

    def test_to_json_creates_parent_directory(self, tmp_path):
        results = _classification_results()
        path = tmp_path / "nested" / "dir" / "cls.json"
        results.to_json(str(path))
        assert path.is_file()

    def test_save_and_load_classification_roundtrip(self, tmp_path):
        results = _classification_results()
        path = tmp_path / "cls.json"
        results.save(str(path))
        loaded = Results.load(str(path))
        assert loaded.task_type == "classification"
        assert loaded.orig_size == (48, 64)
        assert loaded.image_path is None
        np.testing.assert_allclose(loaded.probs, results.probs)
        assert [index for index, _ in loaded.topk] == [1, 2, 0]
        np.testing.assert_allclose([prob for _, prob in loaded.topk], [0.7, 0.2, 0.1])
        assert loaded.meta == {"num_classes": 3}
        assert loaded.to_dict() == results.to_dict()

    def test_save_and_load_detection_roundtrip(self, tmp_path):
        results = _detection_results()
        path = tmp_path / "det.json"
        results.save(str(path))
        loaded = Results.load(str(path))
        np.testing.assert_allclose(loaded.boxes, results.boxes)
        np.testing.assert_allclose(loaded.scores, results.scores)
        np.testing.assert_array_equal(loaded.labels, results.labels)
        assert loaded.image_path == results.image_path
        assert loaded.meta == {"耗时": 0.012, "names": ["cat", "dog"]}
        assert loaded.to_dict() == results.to_dict()

    def test_save_and_load_segmentation_roundtrip(self, tmp_path):
        results = _segmentation_results()
        path = tmp_path / "seg.json"
        results.save(str(path))
        loaded = Results.load(str(path))
        assert loaded.mask.dtype == np.int64
        np.testing.assert_array_equal(loaded.mask, results.mask)
        assert loaded.orig_size == (40, 50)
        assert loaded.to_dict() == results.to_dict()

    def test_empty_detection_roundtrip(self, tmp_path):
        results = Results(
            task_type="detection",
            orig_size=(32, 32),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )
        path = tmp_path / "empty.json"
        results.save(str(path))
        loaded = Results.load(str(path))
        assert loaded.boxes.shape == (0, 4)
        assert loaded.scores.shape == (0,)
        assert loaded.labels.shape == (0,)

    def test_meta_numpy_values_are_jsonable(self, tmp_path):
        results = Results(
            task_type="segmentation",
            orig_size=(10, 12),
            mask=np.zeros((10, 12), dtype=np.int64),
            meta={"probs": np.array([[0.1, 0.9]], dtype=np.float32)},
        )
        path = tmp_path / "meta.json"
        results.save(str(path))
        loaded = Results.load(str(path))
        assert loaded.meta["probs"] == [[pytest.approx(0.1), pytest.approx(0.9)]]

    def test_orig_size_normalized_to_tuple(self):
        results = Results(task_type="classification", orig_size=[20, 30])
        assert results.orig_size == (20, 30)
        assert results.to_dict()["orig_size"] == [20, 30]

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(InferenceError, match="无法读取"):
            Results.load(str(tmp_path / "missing.json"))

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(InferenceError, match="解析失败"):
            Results.load(str(path))

    def test_load_non_object_json_raises(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(InferenceError, match="根节点"):
            Results.load(str(path))

    def test_invalid_orig_size_raises(self):
        with pytest.raises(InferenceError, match="orig_size"):
            Results(task_type="detection", orig_size=(0, 10))


class TestVisualize:
    def test_classification_visualize_shape_dtype(self):
        results = _classification_results()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        visualized = results.visualize(image, class_names=["cat", "dog", "bird"])
        assert visualized.shape == (48, 64, 3)
        assert visualized.dtype == np.uint8
        assert visualized.any()

    def test_classification_visualize_without_image_uses_dark_canvas(self):
        results = _classification_results()
        visualized = results.visualize()
        assert visualized.shape == (48, 64, 3)
        assert visualized.dtype == np.uint8
        assert visualized.any()
        assert int(visualized[47, 63].sum()) > 0  # 深色底也非全黑

    def test_classification_visualize_derives_topk_from_probs(self):
        results = Results(
            task_type="classification",
            orig_size=(32, 48),
            probs=np.array([0.05, 0.15, 0.8], dtype=np.float32),
        )
        visualized = results.visualize(np.zeros((32, 48, 3), dtype=np.uint8))
        assert visualized.any()

    def test_detection_visualize_shape_dtype(self):
        results = _detection_results()
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        visualized = results.visualize(image, class_names=["cat", "dog"])
        assert visualized.shape == (64, 80, 3)
        assert visualized.dtype == np.uint8
        assert visualized.any()

    def test_detection_visualize_without_image_uses_dark_canvas(self):
        results = Results(
            task_type="detection",
            orig_size=(64, 80),
            boxes=np.array([[5, 6, 30, 40]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            labels=np.array([1], dtype=np.int64),
        )
        visualized = results.visualize()
        assert visualized.shape == (64, 80, 3)
        assert visualized.dtype == np.uint8
        assert visualized.any()

    def test_visualize_missing_image_path_raises(self, tmp_path):
        results = _classification_results()
        results.image_path = str(tmp_path / "missing.jpg")
        with pytest.raises(InferenceError, match="无法读取"):
            results.visualize()

    def test_detection_visualize_clips_out_of_bounds_boxes(self):
        results = Results(
            task_type="detection",
            orig_size=(32, 32),
            boxes=np.array([[-100, -100, 1000, 1000]], dtype=np.float32),
            scores=np.array([0.99], dtype=np.float32),
            labels=np.array([3], dtype=np.int64),
        )
        visualized = results.visualize(np.zeros((32, 32, 3), dtype=np.uint8))
        assert visualized.shape == (32, 32, 3)
        assert visualized.any()

    def test_segmentation_visualize_blends_palette(self):
        results = _segmentation_results()
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        visualized = results.visualize(image)
        assert visualized.shape == (40, 50, 3)
        assert visualized.dtype == np.uint8
        assert visualized.any()
        assert not np.array_equal(visualized, image)
        # 三种类别 -> 至少三种不同颜色
        unique_colors = np.unique(visualized.reshape(-1, 3), axis=0)
        assert unique_colors.shape[0] >= 3

    def test_segmentation_visualize_resizes_mask_to_canvas(self):
        mask = np.zeros((10, 12), dtype=np.int64)
        mask[:5] = 1
        results = Results(task_type="segmentation", orig_size=(20, 24), mask=mask)
        visualized = results.visualize(np.zeros((20, 24, 3), dtype=np.uint8))
        assert visualized.shape == (20, 24, 3)
        assert visualized.dtype == np.uint8

    def test_visualize_reads_image_path(self, tmp_path):
        results = _segmentation_results()
        path = tmp_path / "seg.png"
        source = np.zeros((40, 50, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), source)
        results.image_path = str(path)
        visualized = results.visualize()
        assert visualized.shape == (40, 50, 3)
        assert visualized.dtype == np.uint8

    def test_visualize_resizes_base_image_to_orig_size(self):
        results = _classification_results()
        visualized = results.visualize(np.zeros((10, 10, 3), dtype=np.uint8))
        assert visualized.shape == (48, 64, 3)

    def test_visualize_accepts_pil_image(self):
        results = _classification_results()
        visualized = results.visualize(Image.new("RGB", (64, 48), color=(10, 20, 30)))
        assert visualized.shape == (48, 64, 3)
        assert visualized.dtype == np.uint8

    def test_visualize_unknown_task_returns_base_image(self):
        results = Results(task_type="unknown", orig_size=(16, 20))
        visualized = results.visualize(np.zeros((16, 20, 3), dtype=np.uint8))
        assert visualized.shape == (16, 20, 3)
        assert visualized.dtype == np.uint8

    def test_visualize_empty_detection_still_returns_image(self):
        results = Results(
            task_type="detection",
            orig_size=(16, 20),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )
        visualized = results.visualize(np.zeros((16, 20, 3), dtype=np.uint8))
        assert visualized.shape == (16, 20, 3)
        assert visualized.dtype == np.uint8
