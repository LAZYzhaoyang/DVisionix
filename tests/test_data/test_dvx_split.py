# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集切分（SplitResult / build_splits）单元测试。
"""DVX 数据集切分（``SplitResult`` / ``build_splits``）单元测试。

覆盖契约 P1-5：

- ratio：确定性、分层、train 非空、空 split warning、类别缺失 warning；
- dir：目录归属、多任务 task.path、最长目录优先、未覆盖报错；
- list：JSON/YAML 清单、重叠/缺失/未知路径报错；
- 通用：下标不重叠且完整覆盖、save/load、summary、stats 结构。

``SampleRef`` 由后续 agent 的 scan.py 提供，这里用等价的鸭子类型夹具。
"""

import dataclasses
import json
import types

import pytest
import yaml

from dvisionix.data.spec.schema import DVXDatasetSpec, ParseConfig, SplitConfig, TaskSpec
from dvisionix.data.spec.split import SplitResult, build_splits
from dvisionix.errors import SplitError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FakeRef:
    """与 SampleRef 切分相关字段等价的测试夹具。"""

    rel_path: str
    label: int | None = None
    task: str | None = None


def _spec(
    tmp_path,
    data_type="classification",
    classes=("cat", "dog"),
    tasks=None,
    splits=None,
    spec_path=None,
):
    """构造一个不依赖磁盘扫描的 spec。"""
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    layout = (
        "folder_per_class"
        if data_type in ("classification", "multi_task_classification")
        else "labelme"
    )
    task_specs = None
    if tasks:
        task_specs = [
            TaskSpec(name=name, path=path, classes=list(names)) for name, path, names in tasks
        ]
    return DVXDatasetSpec(
        version=1,
        data_type=data_type,
        layout=layout,
        root=str(root),
        classes=None if data_type == "multi_task_classification" else list(classes),
        tasks=task_specs,
        splits=splits,
        parse=ParseConfig(),
        spec_path=str(spec_path or (tmp_path / "dvx_dataset.yaml")),
    )


def _refs(counts, prefix="cls"):
    """按每类样本数生成分类样本（rel_path 有序）。"""
    refs = []
    for label, count in enumerate(counts):
        for index in range(count):
            refs.append(FakeRef(rel_path=f"{prefix}_{label:02d}/{index:04d}.jpg", label=label))
    return refs


def _rel_mapping(result, refs):
    """把 {split: [下标]} 转成 {rel_path: split}，便于跨输入顺序比较。"""
    return {
        refs[index].rel_path: name for name, indices in result.splits.items() for index in indices
    }


def _assert_partition(result, total):
    """断言切分结果互不重叠、完整覆盖、stats 自洽。"""
    seen = []
    for name, indices in result.splits.items():
        assert indices == sorted(indices), f"{name} 下标必须升序"
        assert len(indices) == len(set(indices)), f"{name} 下标不能重复"
        seen.extend(indices)
    assert sorted(seen) == list(range(total))
    assert result.stats["total"] == total
    assert set(result.stats["splits"]) == set(result.splits)
    for name, indices in result.splits.items():
        info = result.stats["splits"][name]
        assert info["count"] == len(indices)
        assert isinstance(info["per_class"], dict)
    assert isinstance(result.stats["warnings"], list)


def _ratio_config(**overrides):
    fields = {
        "mode": "ratio",
        "ratio": {"train": 0.8, "val": 0.2},
        "seed": 42,
        "stratify": "class",
    }
    fields.update(overrides)
    return SplitConfig(**fields)


# ---------------------------------------------------------------------------
# ratio 模式
# ---------------------------------------------------------------------------


class TestRatioMode:
    def test_basic_counts_and_stats(self, tmp_path):
        refs = _refs([20, 10])
        spec = _spec(tmp_path)
        result = build_splits(refs, _ratio_config(stratify="none"), spec)

        _assert_partition(result, len(refs))
        assert result.mode == "ratio"
        assert result.seed == 42
        assert list(result.splits) == ["train", "val"]
        assert len(result.splits["train"]) == 24
        assert len(result.splits["val"]) == 6
        assert result.stats["splits"]["val"]["count"] == 6
        assert sum(result.stats["splits"]["val"]["per_class"].values()) == 6
        assert result.stats["warnings"] == []

    def test_stratified_per_class_counts(self, tmp_path):
        refs = _refs([90, 10])
        spec = _spec(tmp_path)
        result = build_splits(refs, _ratio_config(), spec)

        _assert_partition(result, 100)
        assert result.stats["splits"]["train"]["per_class"] == {"cat": 72, "dog": 8}
        assert result.stats["splits"]["val"]["per_class"] == {"cat": 18, "dog": 2}
        assert result.stats["warnings"] == []

    def test_three_splits(self, tmp_path):
        refs = _refs([25, 25])
        spec = _spec(tmp_path)
        config = _ratio_config(ratio={"train": 0.6, "val": 0.2, "test": 0.2})
        result = build_splits(refs, config, spec)

        _assert_partition(result, 50)
        assert list(result.splits) == ["train", "val", "test"]
        assert [len(result.splits[name]) for name in ("train", "val", "test")] == [30, 10, 10]
        assert result.stats["splits"]["test"]["per_class"] == {"cat": 5, "dog": 5}

    def test_deterministic_same_seed(self, tmp_path):
        refs = _refs([30, 20])
        spec = _spec(tmp_path)
        config = _ratio_config(ratio={"train": 0.7, "val": 0.3})
        first = build_splits(refs, config, spec)
        second = build_splits(refs, config, spec)
        assert first.splits == second.splits
        assert first.stats == second.stats

    def test_deterministic_independent_of_input_order(self, tmp_path):
        refs = _refs([30, 20])
        spec = _spec(tmp_path)
        config = _ratio_config(seed=3)
        forward = build_splits(refs, config, spec)
        backward = build_splits(list(reversed(refs)), config, spec)
        assert _rel_mapping(forward, refs) == _rel_mapping(backward, list(reversed(refs)))

    def test_different_seed_changes_assignment(self, tmp_path):
        refs = _refs([30, 20])
        spec = _spec(tmp_path)
        first = build_splits(refs, _ratio_config(seed=1), spec)
        second = build_splits(refs, _ratio_config(seed=2), spec)
        assert first.splits != second.splits

    def test_unstratified_detection_has_no_per_class(self, tmp_path):
        refs = [FakeRef(rel_path=f"img/{index:03d}.jpg") for index in range(25)]
        spec = _spec(tmp_path, data_type="detection", classes=("a", "b"))
        result = build_splits(refs, _ratio_config(stratify=None), spec)

        _assert_partition(result, 25)
        assert len(result.splits["train"]) == 20
        assert len(result.splits["val"]) == 5
        assert result.stats["splits"]["val"]["per_class"] == {}
        assert result.stats["warnings"] == []

    def test_multitask_stratification(self, tmp_path):
        tasks = [
            ("licence", "T/licence", ["无", "短牌照"]),
            ("logo", "T/logo", ["一体奇瑞标", "嵌入奇瑞标"]),
        ]
        spec = _spec(tmp_path, data_type="multi_task_classification", tasks=tasks)
        refs = []
        for task_name, _path, names in tasks:
            for label in range(len(names)):
                for index in range(25):
                    refs.append(
                        FakeRef(
                            rel_path=f"T/{task_name}/{label}/{index:03d}.jpg",
                            label=label,
                            task=task_name,
                        )
                    )
        result = build_splits(refs, _ratio_config(), spec)

        _assert_partition(result, 100)
        assert result.stats["splits"]["val"]["per_class"] == {
            "licence:无": 5,
            "licence:短牌照": 5,
            "logo:一体奇瑞标": 5,
            "logo:嵌入奇瑞标": 5,
        }
        assert result.stats["warnings"] == []

    def test_unbalanced_class_missing_in_val_warns(self, tmp_path):
        refs = _refs([20, 1])
        spec = _spec(tmp_path)
        result = build_splits(refs, _ratio_config(), spec)

        _assert_partition(result, 21)
        assert len(result.splits["train"]) == 17
        assert len(result.splits["val"]) == 4
        assert any("dog" in warn and "val" in warn for warn in result.stats["warnings"])

    def test_class_missing_in_whole_dataset_warns(self, tmp_path):
        refs = _refs([20, 10])
        spec = _spec(tmp_path, classes=("cat", "dog", "fish"))
        result = build_splits(refs, _ratio_config(), spec)
        assert any("fish" in warn for warn in result.stats["warnings"])

    def test_empty_split_warns(self, tmp_path):
        refs = _refs([10, 10])
        spec = _spec(tmp_path)
        config = _ratio_config(ratio={"train": 1.0, "val": 0.0}, seed=0)
        result = build_splits(refs, config, spec)

        _assert_partition(result, 20)
        assert result.splits["val"] == []
        assert any("val 为空" in warn for warn in result.stats["warnings"])
        assert any("类别 cat 在 val 中没有样本" in warn for warn in result.stats["warnings"])

    def test_train_is_never_empty(self, tmp_path):
        classes = tuple(f"c{index}" for index in range(10))
        refs = [FakeRef(rel_path=f"{index}.jpg", label=index) for index in range(10)]
        spec = _spec(tmp_path, classes=classes)
        config = _ratio_config(ratio={"train": 0.1, "val": 0.9}, seed=0)
        result = build_splits(refs, config, spec)

        _assert_partition(result, 10)
        assert len(result.splits["train"]) == 1
        assert len(result.splits["val"]) == 9
        assert any("train 切分为空" in warn for warn in result.stats["warnings"])

    def test_too_few_samples(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([1]), _ratio_config(), spec)
        assert "少于切分数" in str(excinfo.value)
        assert excinfo.value.hint

    def test_too_few_samples_for_three_way(self, tmp_path):
        spec = _spec(tmp_path)
        config = _ratio_config(ratio={"train": 0.6, "val": 0.2, "test": 0.2})
        with pytest.raises(SplitError):
            build_splits(_refs([2]), config, spec)

    def test_empty_refs(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits([], _ratio_config(), spec)
        assert "没有可切分" in str(excinfo.value)


# ---------------------------------------------------------------------------
# build_splits 的 config 归一化与非法配置
# ---------------------------------------------------------------------------


class TestConfigHandling:
    def test_config_none_uses_spec_splits(self, tmp_path):
        splits = SplitConfig(
            mode="ratio",
            ratio={"train": 0.5, "val": 0.5},
            seed=5,
            stratify="none",
        )
        spec = _spec(tmp_path, splits=splits)
        refs = _refs([10, 10])
        result = build_splits(refs, None, spec)
        assert result.seed == 5
        assert len(result.splits["train"]) == 10
        assert len(result.splits["val"]) == 10

    def test_config_none_without_spec_splits(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), None, spec)
        assert "缺少切分配置" in str(excinfo.value)

    def test_mapping_config_accepted(self, tmp_path):
        spec = _spec(tmp_path)
        config = {
            "mode": "ratio",
            "ratio": {"train": 0.8, "val": 0.2},
            "seed": 11,
            "stratify": "none",
        }
        result = build_splits(_refs([10, 10]), config, spec)
        assert result.seed == 11
        assert len(result.splits["train"]) == 16

    def test_invalid_config_type(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError):
            build_splits(_refs([4, 4]), 42, spec)  # type: ignore[arg-type]

    def test_unknown_mode(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), SplitConfig(mode="random"), spec)
        assert "切分模式" in str(excinfo.value)

    def test_ratio_sum_invalid(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), SplitConfig(ratio={"train": 0.5}), spec)
        assert "之和" in str(excinfo.value)

    def test_ratio_unknown_name(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(ratio={"train": 0.8, "valid": 0.2})
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), config, spec)
        assert "非法切分名" in str(excinfo.value)

    def test_ratio_negative(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(ratio={"train": 1.2, "val": -0.2})
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), config, spec)
        assert "负数" in str(excinfo.value)

    def test_seed_not_int(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(seed="42")  # type: ignore[arg-type]
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), config, spec)
        assert "seed" in str(excinfo.value)

    def test_stratify_invalid(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(stratify="label")
        with pytest.raises(SplitError) as excinfo:
            build_splits(_refs([4, 4]), config, spec)
        assert "stratify" in str(excinfo.value)

    def test_ref_without_rel_path(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(ratio={"train": 1.0}, seed=0, stratify="none")
        with pytest.raises(SplitError) as excinfo:
            build_splits([types.SimpleNamespace()], config, spec)
        assert "rel_path" in str(excinfo.value)


# ---------------------------------------------------------------------------
# dir 模式
# ---------------------------------------------------------------------------


class TestDirMode:
    def test_basic_directory_assignment(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/train/a.jpg", label=0),
            FakeRef(rel_path="train_data/val/b.jpg", label=1),
            FakeRef(rel_path="train_data/train/c.jpg", label=0),
        ]
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data/train", "val": "train_data/val"},
        )
        result = build_splits(refs, config, spec)

        _assert_partition(result, 3)
        assert result.mode == "dir"
        assert result.seed is None
        assert result.splits == {"train": [0, 2], "val": [1]}
        assert result.stats["splits"]["val"]["per_class"] == {"dog": 1}

    def test_multitask_directory_relative_to_task_path(self, tmp_path):
        tasks = [
            ("licence", "T/licence", ["无", "短牌照"]),
            ("logo", "T/logo", ["一体奇瑞标", "嵌入奇瑞标"]),
        ]
        refs = [
            FakeRef(rel_path="T/licence/train_data/train/a.jpg", label=0, task="licence"),
            FakeRef(rel_path="T/licence/train_data/val/b.jpg", label=1, task="licence"),
            FakeRef(rel_path="T/logo/train_data/train/c.jpg", label=0, task="logo"),
            FakeRef(rel_path="T/logo/train_data/val/d.jpg", label=1, task="logo"),
        ]
        spec = _spec(tmp_path, data_type="multi_task_classification", tasks=tasks)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data/train", "val": "train_data/val"},
        )
        result = build_splits(refs, config, spec)

        _assert_partition(result, 4)
        assert result.splits == {"train": [0, 2], "val": [1, 3]}
        assert result.stats["splits"]["train"]["per_class"] == {
            "licence:无": 1,
            "logo:一体奇瑞标": 1,
        }

    def test_uncovered_sample_raises(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/train/a.jpg", label=0),
            FakeRef(rel_path="other/b.jpg", label=1),
        ]
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data/train", "val": "train_data/val"},
        )
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "未被 dir_map 覆盖" in str(excinfo.value)
        assert "other/b.jpg" in str(excinfo.value)
        assert excinfo.value.hint

    def test_longest_directory_wins(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/val/a.jpg", label=0),
            FakeRef(rel_path="train_data/train/b.jpg", label=1),
        ]
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data", "val": "train_data/val"},
        )
        result = build_splits(refs, config, spec)
        assert result.splits == {"train": [1], "val": [0]}
        assert any("同时命中" in warn for warn in result.stats["warnings"])

    def test_empty_split_warns(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/train/a.jpg", label=0),
            FakeRef(rel_path="train_data/val/b.jpg", label=1),
        ]
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={
                "train": "train_data/train",
                "val": "train_data/val",
                "test": "train_data/test",
            },
        )
        result = build_splits(refs, config, spec)
        assert result.splits["test"] == []
        assert result.stats["splits"]["test"]["count"] == 0
        assert any("test 为空" in warn for warn in result.stats["warnings"])

    def test_absolute_directory_rejected(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "/abs/train", "val": "train_data/val"},
        )
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a.jpg", label=0)], config, spec)
        assert "相对目录" in str(excinfo.value)

    def test_missing_dir_map(self, tmp_path):
        spec = _spec(tmp_path)
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a.jpg", label=0)], SplitConfig(mode="dir"), spec)
        assert "dir_map" in str(excinfo.value)

    def test_unknown_split_name(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(mode="dir", dir_map={"train": "t", "valid": "v"})
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a.jpg", label=0)], config, spec)
        assert "非法切分名" in str(excinfo.value)

    def test_missing_train(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(mode="dir", dir_map={"val": "v"})
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a.jpg", label=0)], config, spec)
        assert "train" in str(excinfo.value)

    def test_backslash_and_trailing_slash_normalized(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/train/a.jpg", label=0),
            FakeRef(rel_path="./train_data/val/b.jpg", label=1),
        ]
        spec = _spec(tmp_path)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data\\train", "val": "train_data/val/"},
        )
        result = build_splits(refs, config, spec)
        assert result.splits == {"train": [0], "val": [1]}

    def test_unknown_task_is_uncovered(self, tmp_path):
        tasks = [("licence", "T/licence", ["a"])]
        spec = _spec(tmp_path, data_type="multi_task_classification", tasks=tasks)
        config = SplitConfig(
            mode="dir",
            dir_map={"train": "train_data/train", "val": "train_data/val"},
        )
        refs = [FakeRef(rel_path="T/licence/train_data/train/a.jpg", label=0, task=None)]
        with pytest.raises(SplitError):
            build_splits(refs, config, spec)


# ---------------------------------------------------------------------------
# list 模式
# ---------------------------------------------------------------------------


def _write_json(tmp_path, payload, name="splits.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_yaml_file(tmp_path, payload, name="splits.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return str(path)


class TestListMode:
    def test_basic_json(self, tmp_path):
        refs = [
            FakeRef(rel_path="a/1.jpg", label=0),
            FakeRef(rel_path="a/2.jpg", label=1),
            FakeRef(rel_path="b/1.jpg", label=0),
        ]
        _write_json(tmp_path, {"train": ["a/1.jpg", "b/1.jpg"], "val": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        result = build_splits(refs, config, spec)

        _assert_partition(result, 3)
        assert result.mode == "list"
        assert result.seed is None
        assert result.splits == {"train": [0, 2], "val": [1]}
        assert result.stats["splits"]["train"]["per_class"] == {"cat": 2}
        assert result.stats["splits"]["val"]["per_class"] == {"dog": 1}

    def test_yaml_list_file(self, tmp_path):
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        _write_yaml_file(tmp_path, {"train": ["a/1.jpg"], "val": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        result = build_splits(refs, SplitConfig(mode="list", list_file="splits.yaml"), spec)
        assert result.splits == {"train": [0], "val": [1]}

    def test_list_file_relative_to_spec_path(self, tmp_path):
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        (conf_dir / "splits.json").write_text(
            json.dumps({"train": ["a/1.jpg"], "val": ["a/2.jpg"]}), encoding="utf-8"
        )
        spec = _spec(tmp_path, spec_path=conf_dir / "dvx_dataset.yaml")
        result = build_splits(refs, SplitConfig(mode="list", list_file="splits.json"), spec)
        assert result.splits == {"train": [0], "val": [1]}

    def test_absolute_list_file(self, tmp_path):
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        path = _write_json(tmp_path, {"train": ["a/1.jpg"], "val": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        result = build_splits(refs, SplitConfig(mode="list", list_file=path), spec)
        assert result.splits == {"train": [0], "val": [1]}

    def test_missing_list_file(self, tmp_path):
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="nope.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "清单文件不存在" in str(excinfo.value)

    def test_missing_path(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg"], "val": ["zzz.jpg"]})
        spec = _spec(tmp_path)
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "zzz.jpg" in str(excinfo.value)
        assert "不存在" in str(excinfo.value)

    def test_overlap_between_splits(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg"], "val": ["a/1.jpg"]})
        spec = _spec(tmp_path)
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "同时出现" in str(excinfo.value)

    def test_duplicate_path_in_same_split(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg", "a/1.jpg"], "val": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "重复" in str(excinfo.value)

    def test_unlisted_sample(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg"], "val": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        refs = [
            FakeRef(rel_path="a/1.jpg", label=0),
            FakeRef(rel_path="a/2.jpg", label=1),
            FakeRef(rel_path="a/3.jpg", label=0),
        ]
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "未出现在清单文件" in str(excinfo.value)
        assert "a/3.jpg" in str(excinfo.value)

    def test_missing_train(self, tmp_path):
        _write_json(tmp_path, {"val": ["a/1.jpg"]})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "train" in str(excinfo.value)

    def test_unknown_split_name(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg"], "valid": ["a/2.jpg"]})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "非法切分名" in str(excinfo.value)

    def test_bad_top_level(self, tmp_path):
        _write_json(tmp_path, ["a/1.jpg"])
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "顶层必须是映射" in str(excinfo.value)

    def test_value_not_list(self, tmp_path):
        _write_json(tmp_path, {"train": "a/1.jpg"})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "必须是路径列表" in str(excinfo.value)

    def test_non_string_entry(self, tmp_path):
        _write_json(tmp_path, {"train": [1]})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "非字符串路径" in str(excinfo.value)

    def test_empty_payload(self, tmp_path):
        _write_json(tmp_path, {})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        with pytest.raises(SplitError) as excinfo:
            build_splits([FakeRef(rel_path="a/1.jpg", label=0)], config, spec)
        assert "清单为空" in str(excinfo.value)

    def test_duplicate_ref_rel_path(self, tmp_path):
        _write_json(tmp_path, {"train": ["a/1.jpg"], "val": []})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/1.jpg", label=1)]
        with pytest.raises(SplitError) as excinfo:
            build_splits(refs, config, spec)
        assert "重复 rel_path" in str(excinfo.value)

    def test_empty_test_split_allowed_with_warning(self, tmp_path):
        refs = [FakeRef(rel_path="a/1.jpg", label=0), FakeRef(rel_path="a/2.jpg", label=1)]
        _write_json(tmp_path, {"train": ["a/1.jpg"], "val": ["a/2.jpg"], "test": []})
        spec = _spec(tmp_path)
        config = SplitConfig(mode="list", list_file="splits.json")
        result = build_splits(refs, config, spec)
        assert result.splits == {"train": [0], "val": [1], "test": []}
        assert any("test 为空" in warn for warn in result.stats["warnings"])


# ---------------------------------------------------------------------------
# SplitResult：save / load / summary
# ---------------------------------------------------------------------------


class TestSplitResult:
    def test_save_load_roundtrip(self, tmp_path):
        refs = _refs([10, 10])
        spec = _spec(tmp_path)
        result = build_splits(refs, _ratio_config(), spec)
        target = tmp_path / "out" / "split.json"
        result.save(str(target))

        assert target.is_file()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert set(payload) == {"splits", "stats", "mode", "seed"}
        loaded = SplitResult.load(str(target))
        assert loaded == result
        assert "train" in loaded.summary()

    def test_save_creates_parent_directory(self, tmp_path):
        spec = _spec(tmp_path)
        result = build_splits(_refs([4, 4]), _ratio_config(), spec)
        target = tmp_path / "a" / "b" / "split.json"
        result.save(str(target))
        assert target.is_file()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(SplitError) as excinfo:
            SplitResult.load(str(tmp_path / "nope.json"))
        assert "不存在" in str(excinfo.value)

    def test_load_bad_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SplitError) as excinfo:
            SplitResult.load(str(path))
        assert "JSON" in str(excinfo.value)

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"splits": "bad", "stats": {}, "mode": "ratio"},
            {"splits": {"train": "bad"}, "stats": {}, "mode": "ratio"},
            {"splits": {"train": [0, "x"]}, "stats": {}, "mode": "ratio"},
            {"splits": {"train": [-1]}, "stats": {}, "mode": "ratio"},
            {"splits": {"train": [0]}, "stats": [], "mode": "ratio"},
            {"splits": {"train": [0]}, "stats": {}},
            {"splits": {"train": [0]}, "stats": {}, "mode": 3},
            {"splits": {"train": [0]}, "stats": {}, "mode": "ratio", "seed": "42"},
        ],
    )
    def test_load_bad_structure(self, tmp_path, payload):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SplitError):
            SplitResult.load(str(path))

    def test_load_sorts_indices(self, tmp_path):
        payload = {
            "splits": {"train": [2, 0, 1]},
            "stats": {"total": 3, "splits": {}, "warnings": []},
            "mode": "ratio",
            "seed": None,
        }
        path = tmp_path / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = SplitResult.load(str(path))
        assert loaded.splits == {"train": [0, 1, 2]}
        assert loaded.seed is None

    def test_summary_contains_counts_and_warnings(self, tmp_path):
        spec = _spec(tmp_path)
        config = _ratio_config(ratio={"train": 1.0, "val": 0.0}, seed=0)
        result = build_splits(_refs([5, 5]), config, spec)
        text = result.summary()
        assert "切分模式: ratio" in text
        assert "seed=0" in text
        assert "总样本数: 10" in text
        assert "train: 10 个样本" in text
        assert "val: 0 个样本" in text
        assert "警告" in text

    def test_stats_structure(self, tmp_path):
        result = build_splits(_refs([10, 10]), _ratio_config(), _spec(tmp_path))
        stats = result.stats
        assert set(stats) == {"total", "splits", "warnings"}
        assert stats["total"] == 20
        for info in stats["splits"].values():
            assert set(info) == {"count", "per_class"}
            assert isinstance(info["count"], int)
            assert all(
                isinstance(name, str) and isinstance(count, int)
                for name, count in info["per_class"].items()
            )
        assert isinstance(stats["warnings"], list)
        assert all(isinstance(warn, str) for warn in stats["warnings"])

    def test_dir_and_list_mode_seed_is_none(self, tmp_path):
        refs = [
            FakeRef(rel_path="train_data/train/a.jpg", label=0),
            FakeRef(rel_path="train_data/val/b.jpg", label=1),
        ]
        spec = _spec(tmp_path)
        dir_result = build_splits(
            refs,
            SplitConfig(
                mode="dir",
                dir_map={"train": "train_data/train", "val": "train_data/val"},
            ),
            spec,
        )
        assert dir_result.seed is None

        _write_json(
            tmp_path, {"train": ["train_data/train/a.jpg"], "val": ["train_data/val/b.jpg"]}
        )
        list_result = build_splits(refs, SplitConfig(mode="list", list_file="splits.json"), spec)
        assert list_result.seed is None
        assert list_result.mode == "list"
