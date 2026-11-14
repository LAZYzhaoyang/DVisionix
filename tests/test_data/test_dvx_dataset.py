# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 标准数据集（DVXDataset）单元测试。
"""DVX 标准数据集（``DVXDataset``）单元测试（契约 P1-7 / P1-11）。

覆盖范围：

- 构造：spec 对象 / yaml 路径 / ``Path`` 入参，``scan_samples -> build_splits -> split`` 流程；
- 只读属性：``spec / split / data_type / classes / task_names / num_classes / refs /
  split_result``，以及多任务下 ``classes`` / ``num_classes`` 为 None 的约定；
- 四种 ``data_type`` 的 ``__getitem__`` 字段、dtype、shape：
  classification / multi_task_classification / detection（含空标注）/ segmentation（含空标注）；
- 分割掩码：用 ``polygons_to_mask`` 栅格化、背景 0、后写覆盖先写、``(H, W) int64``；
- 切分：ratio 结果与 ``split_result`` 下标一致、传入 ``split_result`` 时直接使用、
  dir 模式（multicls 公共子目录推断）、``split`` 不存在 / 下标越界报错；
- ``task`` 参数：仅多任务可用、按任务过滤、未知任务报错；
- 复用 BaseDataset 流程：先读图再进 transforms、``_check_sample_keys`` 不把 ``task`` 当未知字段；
- DataLoader 能取 batch（分类默认 collate；检测/分割用 dataset 自带的 collate_fn）。

全部夹具用 ``tmp_path`` + numpy + cv2 现场生成，**不依赖** ``project/data``。
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader

from dvisionix.data import DVXDataset, DVXDatasetSpec, SplitResult, load_spec
from dvisionix.data.base import BaseDataset
from dvisionix.data.sample import Sample
from dvisionix.data.spec import ParseConfig, SplitConfig, TaskSpec, build_splits, scan_samples
from dvisionix.data.transforms import BoxesToTensor, LabelToTensor, ToTensor, TransformPipeline
from dvisionix.errors import DatasetSpecError, SplitError

pytestmark = pytest.mark.unit

IMAGE_SIZE = 8
CLASSES = ["cat", "dog"]


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _write_image(path, height=IMAGE_SIZE, width=IMAGE_SIZE):
    """写一张真实可解码的小图（用 imencode 避免非 ASCII 路径写入失败）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(path.suffix.lower(), image)
    assert ok, f"cv2.imencode 失败: {path}"
    path.write_bytes(buffer.tobytes())
    return path


def _shape(label, shape_type, points):
    """构造一个 labelme shape 字典。"""
    return {"label": label, "shape_type": shape_type, "points": points}


def _rect(label, x1, y1, x2, y2):
    """构造两点矩形。"""
    return _shape(label, "rectangle", [[x1, y1], [x2, y2]])


def _polygon(label, points):
    """构造多边形。"""
    return _shape(label, "polygon", points)


def _write_labelme(path, shapes, height=IMAGE_SIZE, width=IMAGE_SIZE):
    """写一个手写 labelme JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": path.with_suffix(".jpg").name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _make_spec(tmp_path, root, data_type, classes=None, tasks=None, splits=None):
    """构造一个指向 ``root`` 的 spec（yaml 路径只占位，不要求存在）。"""
    layout = (
        "folder_per_class"
        if data_type in ("classification", "multi_task_classification")
        else "labelme"
    )
    return DVXDatasetSpec(
        version=1,
        data_type=data_type,
        layout=layout,
        root=str(root),
        classes=list(classes) if classes is not None else None,
        tasks=tasks,
        splits=splits,
        parse=ParseConfig(),
        spec_path=str(tmp_path / "dvx_dataset.yaml"),
    )


def _ratio_config(stratify="class"):
    """0.8 / 0.2 的 ratio 切分配置。"""
    return SplitConfig(
        mode="ratio",
        ratio={"train": 0.8, "val": 0.2},
        seed=42,
        stratify=stratify,
    )


def _all_in_split(count, split="train"):
    """把 ``count`` 个样本全部放进同一个 split（便于确定性断言每条样本）。"""
    return SplitResult(
        splits={split: list(range(count))},
        stats={
            "total": count,
            "splits": {split: {"count": count, "per_class": {}}},
            "warnings": [],
        },
        mode="list",
        seed=None,
    )


def _sample_by_rel_path(dataset, rel_path):
    """按 ``rel_path`` 找到样本并取 ``__getitem__`` 结果。"""
    for index, ref in enumerate(dataset.refs):
        if ref.rel_path == rel_path:
            return dataset[index]
    raise AssertionError(f"{rel_path} 不在当前 split: {[ref.rel_path for ref in dataset.refs]}")


# ---------------------------------------------------------------------------
# 夹具：四种 data_type 各一个迷你数据集
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_spec(tmp_path):
    """10 张图 / 2 类（每类 5 张），ratio 0.8/0.2 分层切分。"""
    root = tmp_path / "data"
    for index in range(5):
        _write_image(root / "cat" / f"cat_{index}.jpg")
        _write_image(root / "dog" / f"dog_{index}.jpg")
    return _make_spec(
        tmp_path,
        root,
        "classification",
        classes=CLASSES,
        splits=_ratio_config("class"),
    )


@pytest.fixture()
def multitask_spec(tmp_path):
    """两个任务：licence 2 类各 3 张，logo 1 类 6 张，共 12 张。"""
    root = tmp_path / "data"
    for index in range(3):
        _write_image(root / "T_licence" / "无" / f"wu_{index}.jpg")
        _write_image(root / "T_licence" / "短牌照" / f"duan_{index}.jpg")
    for index in range(6):
        _write_image(root / "T_logo" / "一体奇瑞标" / f"logo_{index}.jpg")
    tasks = [
        TaskSpec(name="licence", path="T_licence", classes=["无", "短牌照"]),
        TaskSpec(name="logo", path="T_logo", classes=["一体奇瑞标"]),
    ]
    return _make_spec(
        tmp_path,
        root,
        "multi_task_classification",
        tasks=tasks,
        splits=_ratio_config("class"),
    )


@pytest.fixture()
def detection_spec(tmp_path):
    """4 张图：2 框 / 1 框 / 空标注 / 1 框，ratio 0.8/0.2 不分层。"""
    root = tmp_path / "data"
    _write_image(root / "A" / "a.jpg")
    _write_labelme(root / "A" / "a.json", [_rect("cat", 1, 1, 4, 4), _rect("dog", 2, 2, 6, 6)])
    _write_image(root / "A" / "b.jpg")
    _write_labelme(root / "A" / "b.json", [_rect("cat", 1, 2, 5, 6)])
    _write_image(root / "A" / "c.jpg")
    _write_labelme(root / "A" / "c.json", [])
    _write_image(root / "B" / "d.jpg")
    _write_labelme(root / "B" / "d.json", [_rect("dog", 1, 1, 3, 3)])
    return _make_spec(
        tmp_path,
        root,
        "detection",
        classes=CLASSES,
        splits=_ratio_config("none"),
    )


@pytest.fixture()
def segmentation_spec(tmp_path):
    """4 张图：双多边形 / 满图多边形 / 空标注 / 只有矩形（被跳过）。"""
    root = tmp_path / "data"
    _write_image(root / "A" / "a.jpg")
    _write_labelme(
        root / "A" / "a.json",
        [
            _polygon("cat", [[1, 1], [6, 1], [6, 6], [1, 6]]),
            # 后写覆盖先写：中心像素应属于 dog
            _polygon("dog", [[2, 2], [4, 2], [4, 4], [2, 4]]),
        ],
    )
    _write_image(root / "A" / "b.jpg")
    _write_labelme(root / "A" / "b.json", [_polygon("cat", [[0, 0], [7, 0], [7, 7], [0, 7]])])
    _write_image(root / "A" / "c.jpg")
    _write_labelme(root / "A" / "c.json", [])
    _write_image(root / "B" / "d.jpg")
    _write_labelme(root / "B" / "d.json", [_rect("cat", 1, 1, 4, 4)])
    return _make_spec(
        tmp_path,
        root,
        "segmentation",
        classes=["__background__", "cat", "dog"],
        splits=_ratio_config("none"),
    )


@pytest.fixture()
def multicls_dir_spec(tmp_path):
    """multicls 布局：扁平类目录 + train_data/{train,val} 重复，dir 模式只扫公共子目录。"""
    root = tmp_path / "data"
    _write_image(root / "T_licence" / "无" / "flat.jpg")
    _write_image(root / "T_licence" / "train_data" / "train" / "无" / "train.jpg")
    _write_image(root / "T_licence" / "train_data" / "val" / "无" / "val.jpg")
    _write_image(root / "T_logo" / "一体奇瑞标" / "flat.jpg")
    _write_image(root / "T_logo" / "train_data" / "train" / "一体奇瑞标" / "train.jpg")
    _write_image(root / "T_logo" / "train_data" / "val" / "一体奇瑞标" / "val.jpg")
    tasks = [
        TaskSpec(name="licence", path="T_licence", classes=["无"]),
        TaskSpec(name="logo", path="T_logo", classes=["一体奇瑞标"]),
    ]
    splits = SplitConfig(
        mode="dir",
        dir_map={"train": "train_data/train", "val": "train_data/val"},
    )
    return _make_spec(tmp_path, root, "multi_task_classification", tasks=tasks, splits=splits)


# ---------------------------------------------------------------------------
# 只读属性
# ---------------------------------------------------------------------------


class TestProperties:
    def test_classification_properties(self, classification_spec):
        dataset = DVXDataset(classification_spec)

        assert isinstance(dataset, BaseDataset)
        assert dataset.spec is classification_spec
        assert dataset.split == "train"
        assert dataset.data_type == "classification"
        assert dataset.classes == ["cat", "dog"]
        assert dataset.task_names is None
        assert dataset.num_classes == 2
        assert dataset.split_result.mode == "ratio"
        assert dataset.split_result.seed == 42
        assert len(dataset.refs) == len(dataset) == 8
        assert all(ref.task is None for ref in dataset.refs)
        assert not hasattr(dataset, "task_type") or dataset.task_type == "classification"

    def test_multitask_properties(self, multitask_spec):
        dataset = DVXDataset(multitask_spec)

        assert dataset.data_type == "multi_task_classification"
        assert dataset.task_names == ["licence", "logo"]
        # 多任务各任务有独立类别表，没有统一的 classes / num_classes
        assert dataset.classes is None
        assert dataset.num_classes is None
        assert len(dataset.refs) == len(dataset.split_result.splits["train"])
        assert len(dataset.refs) == 9
        assert sum(len(indices) for indices in dataset.split_result.splits.values()) == 12
        assert {ref.task for ref in dataset.refs} == {"licence", "logo"}

    def test_detection_properties(self, detection_spec):
        dataset = DVXDataset(detection_spec)

        assert dataset.data_type == "detection"
        assert dataset.classes == CLASSES
        assert dataset.task_names is None
        assert dataset.num_classes == 2
        assert len(dataset.refs) == len(dataset.split_result.splits["train"]) == 3
        assert sum(len(indices) for indices in dataset.split_result.splits.values()) == 4
        assert dataset.collate_fn is not None

    def test_segmentation_properties(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec)

        assert dataset.data_type == "segmentation"
        assert dataset.classes == ["__background__", "cat", "dog"]
        assert dataset.num_classes == 3
        assert dataset.task_names is None
        assert len(dataset.refs) == len(dataset.split_result.splits["train"]) == 3
        assert sum(len(indices) for indices in dataset.split_result.splits.values()) == 4
        assert dataset.collate_fn is not None

    def test_properties_are_read_only(self, classification_spec):
        dataset = DVXDataset(classification_spec)

        for name in (
            "spec",
            "split",
            "data_type",
            "classes",
            "task_names",
            "num_classes",
            "refs",
            "split_result",
        ):
            with pytest.raises(AttributeError):
                setattr(dataset, name, None)

    def test_refs_and_classes_return_copies(self, classification_spec):
        dataset = DVXDataset(classification_spec)

        dataset.refs.clear()
        dataset.classes.append("bird")

        assert len(dataset.refs) == 8
        assert dataset.classes == ["cat", "dog"]


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------


class TestSplits:
    def test_ratio_split_matches_split_result(self, classification_spec):
        all_refs = scan_samples(classification_spec)
        dataset = DVXDataset(classification_spec)
        result = dataset.split_result

        train_indices = set(result.splits["train"])
        val_indices = set(result.splits["val"])
        assert train_indices.isdisjoint(val_indices)
        assert train_indices | val_indices == set(range(len(all_refs)))
        expected = [all_refs[index].rel_path for index in result.splits["train"]]
        assert [ref.rel_path for ref in dataset.refs] == expected
        assert len(dataset) == 8

        val = DVXDataset(classification_spec, split="val")
        assert len(val) == 2
        assert {ref.rel_path for ref in dataset.refs}.isdisjoint({ref.rel_path for ref in val.refs})
        assert len(dataset) + len(val) == len(all_refs) == 10

    def test_ratio_split_is_deterministic(self, classification_spec):
        first = DVXDataset(classification_spec)
        second = DVXDataset(classification_spec)

        assert [ref.rel_path for ref in first.refs] == [ref.rel_path for ref in second.refs]
        assert first.split_result.splits == second.split_result.splits

    def test_explicit_split_result_is_used_directly(self, classification_spec):
        all_refs = scan_samples(classification_spec)
        result = SplitResult(
            splits={"train": [0, 2], "val": [1]},
            stats={"total": 3, "splits": {}, "warnings": []},
            mode="list",
            seed=None,
        )

        dataset = DVXDataset(classification_spec, split="train", split_result=result)

        assert dataset.split_result is result
        assert [ref.rel_path for ref in dataset.refs] == [
            all_refs[0].rel_path,
            all_refs[2].rel_path,
        ]
        assert len(dataset) == 2

    def test_build_splits_is_not_called_when_split_result_given(
        self, classification_spec, monkeypatch
    ):
        result = SplitResult(splits={"train": [0]}, stats={}, mode="list", seed=None)
        monkeypatch.setattr(
            "dvisionix.data.dvx_dataset.build_splits",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应构建切分")),
        )

        dataset = DVXDataset(classification_spec, split_result=result)

        assert len(dataset) == 1

    def test_dir_mode_scans_common_subdir(self, multicls_dir_spec, capsys):
        train = DVXDataset(multicls_dir_spec)
        val = DVXDataset(multicls_dir_spec, split="val")
        captured = capsys.readouterr()

        assert len(train) == 2 and len(val) == 2
        assert all("train_data/train/" in ref.rel_path for ref in train.refs)
        assert all("train_data/val/" in ref.rel_path for ref in val.refs)
        assert "train_data" in captured.err  # 自动推断必须显式打印

    def test_unknown_split_raises(self, classification_spec):
        with pytest.raises(SplitError, match="test"):
            DVXDataset(classification_spec, split="test")

    def test_empty_split_name_raises(self, classification_spec):
        with pytest.raises(SplitError, match="split"):
            DVXDataset(classification_spec, split="")

    def test_split_result_out_of_range_raises(self, classification_spec):
        result = SplitResult(splits={"train": [99]}, stats={}, mode="list", seed=None)

        with pytest.raises(SplitError, match="越界"):
            DVXDataset(classification_spec, split_result=result)

    def test_split_result_duplicate_index_raises(self, classification_spec):
        result = SplitResult(splits={"train": [0, 0]}, stats={}, mode="list", seed=None)

        with pytest.raises(SplitError, match="重复"):
            DVXDataset(classification_spec, split_result=result)

    def test_missing_split_config_raises(self, tmp_path):
        root = tmp_path / "data"
        _write_image(root / "cat" / "a.jpg")
        spec = _make_spec(tmp_path, root, "classification", classes=["cat"], splits=None)

        with pytest.raises(SplitError, match="切分配置"):
            DVXDataset(spec)

    def test_build_splits_result_accepted(self, classification_spec):
        refs = scan_samples(classification_spec)
        result = build_splits(refs, classification_spec.splits, classification_spec)
        dataset = DVXDataset(classification_spec, split="val", split_result=result)

        assert dataset.split_result is result
        assert len(dataset) == len(result.splits["val"])


# ---------------------------------------------------------------------------
# 样本内容：classification / multi_task / detection / segmentation
# ---------------------------------------------------------------------------


class TestSampleContent:
    def test_classification_sample(self, classification_spec):
        all_refs = scan_samples(classification_spec)
        dataset = DVXDataset(classification_spec, split_result=_all_in_split(10))

        for index, ref in enumerate(dataset.refs):
            sample = dataset[index]

            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], np.ndarray)
            assert sample["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
            assert sample["image"].dtype == np.uint8
            assert isinstance(sample["label"], int)
            assert sample["label"] == ref.label == all_refs[index].label
            assert Sample(sample).unknown_keys() == []

    def test_multitask_sample(self, multitask_spec):
        dataset = DVXDataset(multitask_spec, split_result=_all_in_split(12))

        tasks = set()
        for index, ref in enumerate(dataset.refs):
            sample = dataset[index]

            assert set(sample) == {"image", "label", "task"}
            assert isinstance(sample["image"], np.ndarray)
            assert isinstance(sample["label"], int)
            assert sample["label"] == ref.label
            assert sample["task"] == ref.task
            assert Sample(sample).unknown_keys() == []  # "task" 已在 EXTENDED_KEYS 中
            tasks.add(sample["task"])

        assert tasks == {"licence", "logo"}

    def test_detection_sample(self, detection_spec):
        all_refs = scan_samples(detection_spec)
        dataset = DVXDataset(detection_spec, split_result=_all_in_split(4))

        for index, ref in enumerate(dataset.refs):
            sample = dataset[index]

            assert set(sample) == {"image", "boxes", "labels"}
            assert isinstance(sample["image"], np.ndarray)
            assert sample["boxes"].dtype == np.float32
            assert sample["boxes"].ndim == 2 and sample["boxes"].shape[1] == 4
            assert sample["labels"].dtype == np.int64
            assert sample["labels"].ndim == 1
            assert sample["boxes"].shape[0] == sample["labels"].shape[0] == len(ref.labels)
            np.testing.assert_array_equal(sample["labels"], all_refs[index].labels)

    def test_detection_boxes_values(self, detection_spec):
        dataset = DVXDataset(detection_spec, split_result=_all_in_split(4))

        np.testing.assert_allclose(
            _sample_by_rel_path(dataset, "A/a.jpg")["boxes"],
            [[1, 1, 4, 4], [2, 2, 6, 6]],
        )

    def test_detection_empty_annotation_is_negative_sample(self, detection_spec):
        dataset = DVXDataset(detection_spec, split_result=_all_in_split(4))
        sample = _sample_by_rel_path(dataset, "A/c.jpg")

        assert sample["boxes"].shape == (0, 4)
        assert sample["boxes"].dtype == np.float32
        assert sample["labels"].shape == (0,)
        assert sample["labels"].dtype == np.int64

    def test_segmentation_mask_shape_and_dtype(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec, split_result=_all_in_split(4))

        for index in range(len(dataset)):
            sample = dataset[index]

            assert set(sample) == {"image", "mask"}
            assert isinstance(sample["image"], np.ndarray)
            assert sample["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
            assert sample["mask"].dtype == np.int64

    def test_segmentation_mask_values(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec, split_result=_all_in_split(4))

        sample = _sample_by_rel_path(dataset, "A/a.jpg")
        # 类别表 [__background__, cat, dog] -> cat=1, dog=2；背景 0
        assert sample["mask"][0, 0] == 0
        assert sample["mask"][1, 1] == 1
        assert sample["mask"][3, 3] == 2  # 后写的 dog 覆盖先写的 cat

    def test_segmentation_empty_annotation_is_background(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec, split_result=_all_in_split(4))

        empty = _sample_by_rel_path(dataset, "A/c.jpg")
        assert np.all(empty["mask"] == 0)

        # 只有 rectangle 的样本对 segmentation 而言同样是空标注（形状被跳过）
        skipped = _sample_by_rel_path(dataset, "B/d.jpg")
        assert np.all(skipped["mask"] == 0)

    def test_segmentation_full_image_polygon(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec, split_result=_all_in_split(4))
        sample = _sample_by_rel_path(dataset, "A/b.jpg")

        assert np.all(sample["mask"] == 1)

    def test_sample_extended_keys_contains_task(self):
        assert "task" in Sample.EXTENDED_KEYS
        assert "task" not in Sample({"image": "x.jpg", "task": "licence"}).unknown_keys()


# ---------------------------------------------------------------------------
# task 参数（仅多任务分类）
# ---------------------------------------------------------------------------


class TestTaskFilter:
    def test_task_filters_samples(self, multitask_spec):
        dataset = DVXDataset(
            multitask_spec,
            split_result=_all_in_split(12),
            task="licence",
        )

        assert len(dataset) == 6
        assert {ref.task for ref in dataset.refs} == {"licence"}
        assert {sample["task"] for sample in (dataset[i] for i in range(len(dataset)))} == {
            "licence"
        }
        # 属性仍描述 spec 层信息
        assert dataset.task_names == ["licence", "logo"]
        assert dataset.classes is None

    def test_task_on_single_label_dataset_raises(self, classification_spec):
        with pytest.raises(DatasetSpecError, match="multi_task_classification"):
            DVXDataset(classification_spec, task="licence")

    def test_unknown_task_raises(self, multitask_spec):
        with pytest.raises(DatasetSpecError, match="未知任务"):
            DVXDataset(multitask_spec, task="engine")


# ---------------------------------------------------------------------------
# 复用 BaseDataset 的读图 / transforms 流程
# ---------------------------------------------------------------------------


class TestBaseFlow:
    def test_transforms_run_after_image_loaded(self, classification_spec):
        seen = []

        def recorder(sample):
            seen.append((sample["image"].shape, sample["image"].dtype))
            sample["image"] = sample["image"][:1, :1].copy()
            return sample

        dataset = DVXDataset(
            classification_spec,
            split_result=_all_in_split(10),
            transforms=recorder,
        )
        sample = dataset[0]

        assert seen == [((IMAGE_SIZE, IMAGE_SIZE, 3), np.dtype(np.uint8))]
        assert sample["image"].shape == (1, 1, 3)

    def test_transform_pipeline_tensorizes(self, classification_spec):
        pipeline = TransformPipeline([ToTensor(keys=("image",)), LabelToTensor()])
        dataset = DVXDataset(
            classification_spec,
            split_result=_all_in_split(10),
            transforms=pipeline,
        )
        sample = dataset[0]

        assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert sample["image"].dtype.is_floating_point
        assert sample["label"].dtype.__str__() == "torch.int64"

    def test_repeated_access_returns_fresh_mask(self, segmentation_spec):
        dataset = DVXDataset(segmentation_spec, split_result=_all_in_split(4))

        first = _sample_by_rel_path(dataset, "A/a.jpg")["mask"]
        second = _sample_by_rel_path(dataset, "A/a.jpg")["mask"]

        assert first is not second
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# DataLoader 取 batch
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_classification_dataloader_batch(self, classification_spec):
        pipeline = TransformPipeline([ToTensor(keys=("image",)), LabelToTensor()])
        dataset = DVXDataset(classification_spec, transforms=pipeline)
        loader = DataLoader(dataset, batch_size=2)

        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["label"].shape == (2,)
        assert batch["label"].dtype.__str__() == "torch.int64"

    def test_multitask_dataloader_batch(self, multitask_spec):
        pipeline = TransformPipeline([ToTensor(keys=("image",)), LabelToTensor()])
        dataset = DVXDataset(multitask_spec, transforms=pipeline)
        loader = DataLoader(dataset, batch_size=2)

        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(batch["task"], list) and len(batch["task"]) == 2
        assert set(batch["task"]).issubset({"licence", "logo"})

    def test_detection_dataloader_batch(self, detection_spec):
        pipeline = TransformPipeline([ToTensor(keys=("image",)), BoxesToTensor()])
        dataset = DVXDataset(detection_spec, transforms=pipeline)
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
        assert batch["boxes"][0].dtype.is_floating_point

    def test_segmentation_dataloader_batch(self, segmentation_spec):
        pipeline = TransformPipeline([ToTensor(keys=("image", "mask"))])
        dataset = DVXDataset(segmentation_spec, transforms=pipeline)
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["mask"].shape == (2, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["mask"].dtype.__str__() == "torch.int64"


# ---------------------------------------------------------------------------
# spec 入参与错误
# ---------------------------------------------------------------------------


class TestSpecInput:
    def test_construct_from_yaml_path(self, tmp_path):
        root = tmp_path / "data"
        for index in range(5):
            _write_image(root / "cat" / f"cat_{index}.jpg")
            _write_image(root / "dog" / f"dog_{index}.jpg")
        yaml_path = tmp_path / "dvx_dataset.yaml"
        yaml_path.write_text(
            "version: 1\n"
            "data_type: classification\n"
            "layout: folder_per_class\n"
            "root: data\n"
            "classes: [cat, dog]\n"
            "splits:\n"
            "  mode: ratio\n"
            "  ratio: {train: 0.8, val: 0.2}\n"
            "  seed: 42\n"
            "  stratify: class\n",
            encoding="utf-8",
        )

        dataset = DVXDataset(str(yaml_path), split="train")

        assert isinstance(dataset.spec, DVXDatasetSpec)
        assert dataset.spec.spec_path == str(yaml_path.resolve())
        assert dataset.spec.root == str(root.resolve())
        assert len(dataset) == 8

        from_path = DVXDataset(Path(yaml_path), split="val")
        assert len(from_path) == 2
        assert load_spec(str(yaml_path)).class_names() == ["cat", "dog"]

    def test_invalid_spec_type_raises(self):
        with pytest.raises(DatasetSpecError, match="spec 必须是"):
            DVXDataset(123)

    def test_empty_dataset_scan_raises(self, tmp_path):
        root = tmp_path / "data"
        (root / "cat").mkdir(parents=True)
        spec = _make_spec(tmp_path, root, "classification", classes=["cat"], splits=_ratio_config())

        with pytest.raises(SplitError, match="没有可切分的样本"):
            DVXDataset(spec)
