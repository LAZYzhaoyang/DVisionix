# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集规格（dvx_dataset.yaml）schema 与校验测试。
"""DVX 数据集规格（``dvx_dataset.yaml``）schema 与校验测试。

覆盖契约 P1-2 的 6 条校验规则、root 解析、多任务校验、parse 归一化与
DVXDatasetSpec 的 class_names / num_classes / resolve_root 接口。
全部夹具用 ``tmp_path`` 现造，不依赖 ``project/data``。
"""

import os

import pytest
import yaml

from dvisionix.data.spec.schema import (
    DVXDatasetSpec,
    ParseConfig,
    SplitConfig,
    TaskSpec,
    load_spec,
)
from dvisionix.errors import DatasetSpecError

pytestmark = pytest.mark.unit

#: 用于区分"未传 classes"与"显式传 classes=None"
_UNSET = object()


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, payload, name="dvx_dataset.yaml"):
    """把 payload 写成 yaml 文件并返回路径。"""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(path)


def _payload(
    tmp_path,
    data_type="classification",
    layout="folder_per_class",
    classes=_UNSET,
    splits=None,
    **overrides,
):
    """构造一份合法 spec payload（可按需覆盖字段）。"""
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    if classes is _UNSET:
        classes = ("cat", "dog")
    if splits is None:
        splits = {"mode": "ratio", "ratio": {"train": 0.8, "val": 0.2}, "seed": 42}
    payload = {
        "version": 1,
        "data_type": data_type,
        "layout": layout,
        "root": str(root),
        "classes": list(classes) if classes is not None else None,
        "splits": splits,
    }
    payload.update(overrides)
    return payload


def _load_payload(tmp_path, payload, name="dvx_dataset.yaml"):
    """写 yaml 并 load_spec。"""
    return load_spec(_write_yaml(tmp_path, payload, name=name))


def _assert_spec_error(tmp_path, payload, match=None):
    """断言 load_spec 抛 DatasetSpecError，且消息/提示符合预期。"""
    with pytest.raises(DatasetSpecError) as excinfo:
        _load_payload(tmp_path, payload)
    text = str(excinfo.value)
    if match:
        assert match in text
    assert excinfo.value.hint, "DatasetSpecError 必须带 hint"
    return excinfo.value


# ---------------------------------------------------------------------------
# 合法 spec 的解析
# ---------------------------------------------------------------------------


class TestLoadSpec:
    def test_classification_spec_fields(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={
                "mode": "ratio",
                "ratio": {"train": 0.8, "val": 0.2},
                "seed": 42,
                "stratify": "class",
            },
        )
        path = _write_yaml(tmp_path, payload)
        spec = load_spec(path)

        assert isinstance(spec, DVXDatasetSpec)
        assert spec.version == 1
        assert spec.data_type == "classification"
        assert spec.layout == "folder_per_class"
        assert spec.root == str(tmp_path / "data")
        assert spec.classes == ["cat", "dog"]
        assert spec.tasks is None
        assert spec.seg_mode == "semantic"
        assert spec.ignore_index == 255
        assert spec.spec_path == path
        assert spec.class_names() == ["cat", "dog"]
        assert spec.num_classes() == 2
        assert spec.resolve_root() == str(tmp_path / "data")
        assert isinstance(spec.splits, SplitConfig)
        assert spec.splits.mode == "ratio"
        assert spec.splits.ratio == {"train": 0.8, "val": 0.2}
        assert spec.splits.seed == 42
        assert spec.splits.stratify == "class"
        assert spec.splits.list_file is None
        assert spec.splits.dir_map is None

    def test_parse_defaults(self, tmp_path):
        spec = _load_payload(tmp_path, _payload(tmp_path))
        assert spec.parse == ParseConfig()
        assert spec.parse.recursive is True
        assert spec.parse.image_ext == (".jpg", ".jpeg", ".png", ".bmp")
        assert spec.parse.labelme_ignore_labels == ("__ignore__", "__background__")
        assert spec.parse.labelme_image_data == "ignore"
        assert spec.parse.labelme_shape_types == {
            "rectangle": "detection",
            "polygon": "segmentation",
        }

    def test_parse_overrides_and_normalization(self, tmp_path):
        payload = _payload(
            tmp_path,
            parse={
                "recursive": False,
                "image_ext": [".JPG", "png"],
                "labelme": {
                    "ignore_labels": ["__ignore__"],
                    "image_data": "error",
                    "shape_types": {"rectangle": "detection"},
                },
            },
        )
        with pytest.warns(UserWarning, match="前导点"):
            spec = _load_payload(tmp_path, payload)
        assert spec.parse.recursive is False
        assert spec.parse.image_ext == (".jpg", ".png")
        assert spec.parse.labelme_ignore_labels == ("__ignore__",)
        assert spec.parse.labelme_image_data == "error"
        assert spec.parse.labelme_shape_types == {"rectangle": "detection"}

    def test_no_optional_sections(self, tmp_path):
        payload = {
            "version": 1,
            "data_type": "classification",
            "layout": "folder_per_class",
            "root": str(tmp_path / "data"),
            "classes": ["cat"],
        }
        (tmp_path / "data").mkdir()
        spec = _load_payload(tmp_path, payload)
        assert spec.splits is None
        assert spec.parse == ParseConfig()
        assert spec.seg_mode == "semantic"
        assert spec.ignore_index == 255
        assert spec.tasks is None
        assert spec.class_names() == ["cat"]

    def test_root_relative_to_yaml_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        payload = _payload(tmp_path)
        payload["root"] = "../data"
        path = _write_yaml(conf_dir, payload)
        spec = load_spec(path)
        assert spec.root == str(data_dir)
        assert spec.resolve_root() == str(data_dir)

    def test_root_absolute_path(self, tmp_path):
        absolute = tmp_path / "somewhere"
        absolute.mkdir()
        payload = _payload(tmp_path)
        payload["root"] = str(absolute)
        spec = _load_payload(tmp_path, payload)
        assert spec.root == str(absolute)

    def test_chinese_class_names_keep_order(self, tmp_path):
        names = ["幻影灰", "新卡其白", "新碳晶黑", "科技灰", "纳斯达克银"]
        payload = _payload(tmp_path, classes=names)
        spec = _load_payload(tmp_path, payload)
        assert spec.classes == names
        assert spec.class_names() == names
        assert spec.num_classes() == 5

    def test_detection_default_stratify_is_none(self, tmp_path):
        payload = _payload(tmp_path, data_type="detection", layout="labelme")
        spec = _load_payload(tmp_path, payload)
        assert spec.splits.stratify == "none"
        assert spec.splits.ratio == {"train": 0.8, "val": 0.2}

    def test_segmentation_with_query_mode(self, tmp_path):
        payload = _payload(
            tmp_path,
            data_type="segmentation",
            layout="labelme",
            classes=["__background__", "road"],
            seg_mode="query",
            ignore_index=254,
        )
        spec = _load_payload(tmp_path, payload)
        assert spec.seg_mode == "query"
        assert spec.ignore_index == 254
        assert spec.class_names() == ["__background__", "road"]
        assert spec.splits.stratify == "none"

    def test_multitask_spec(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification")
        payload["classes"] = None
        payload["tasks"] = [
            {
                "name": "licence",
                "path": "T15C_1CCD6_licence",
                "classes": ["无", "短牌照", "长牌照"],
            },
            {"name": "logo", "path": "T15C_1CCD6_logo", "classes": ["一体奇瑞标", "嵌入奇瑞标"]},
        ]
        spec = _load_payload(tmp_path, payload)

        assert spec.classes is None
        assert isinstance(spec.tasks, list)
        assert [task.name for task in spec.tasks] == ["licence", "logo"]
        assert spec.tasks[0] == TaskSpec(
            name="licence", path="T15C_1CCD6_licence", classes=["无", "短牌照", "长牌照"]
        )
        assert spec.class_names("licence") == ["无", "短牌照", "长牌照"]
        assert spec.class_names("logo") == ["一体奇瑞标", "嵌入奇瑞标"]
        assert spec.num_classes("logo") == 2
        assert spec.splits.stratify == "class"

    def test_class_names_returns_copy(self, tmp_path):
        spec = _load_payload(tmp_path, _payload(tmp_path))
        names = spec.class_names()
        names.append("fish")
        assert spec.class_names() == ["cat", "dog"]

    def test_splits_three_way_ratio(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={
                "mode": "ratio",
                "ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
                "seed": 7,
            },
        )
        spec = _load_payload(tmp_path, payload)
        assert spec.splits.ratio == {"train": 0.6, "val": 0.2, "test": 0.2}
        assert spec.splits.stratify == "class"

    def test_list_mode_resolves_list_file(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={"mode": "list", "list_file": "splits.json", "seed": 1},
        )
        spec = _load_payload(tmp_path, payload)
        assert spec.splits.list_file == str(tmp_path / "splits.json")

    def test_list_mode_absolute_list_file(self, tmp_path):
        target = tmp_path / "meta" / "splits.yaml"
        payload = _payload(
            tmp_path,
            splits={"mode": "list", "list_file": str(target)},
        )
        spec = _load_payload(tmp_path, payload)
        assert spec.splits.list_file == str(target)

    def test_dir_mode_dir_map(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={
                "mode": "dir",
                "dir_map": {"train": "train_data/train", "val": "train_data/val"},
            },
        )
        spec = _load_payload(tmp_path, payload)
        assert spec.splits.mode == "dir"
        assert spec.splits.dir_map == {
            "train": "train_data/train",
            "val": "train_data/val",
        }

    def test_unknown_key_only_warns(self, tmp_path):
        payload = _payload(tmp_path, description="draft spec")
        with pytest.warns(UserWarning, match="未知字段"):
            spec = _load_payload(tmp_path, payload)
        assert spec.class_names() == ["cat", "dog"]

    def test_unknown_splits_key_warns(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={
                "mode": "ratio",
                "ratio": {"train": 0.8, "val": 0.2},
                "extra": True,
            },
        )
        with pytest.warns(UserWarning, match="未知字段"):
            spec = _load_payload(tmp_path, payload)
        assert spec.splits.mode == "ratio"

    def test_load_spec_is_from_yaml_alias(self, tmp_path):
        payload = _payload(tmp_path)
        path = _write_yaml(tmp_path, payload)
        assert load_spec(path).root == DVXDatasetSpec.from_yaml(path).root


# ---------------------------------------------------------------------------
# 失败路径：文件与 YAML 本身
# ---------------------------------------------------------------------------


class TestFileErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetSpecError) as excinfo:
            load_spec(str(tmp_path / "nope.yaml"))
        assert "不存在" in str(excinfo.value)
        assert excinfo.value.hint

    def test_invalid_yaml_syntax(self, tmp_path):
        path = tmp_path / "dvx_dataset.yaml"
        path.write_text("version: [1\n", encoding="utf-8")
        with pytest.raises(DatasetSpecError) as excinfo:
            load_spec(str(path))
        assert "YAML" in str(excinfo.value)
        assert excinfo.value.hint

    def test_top_level_not_mapping(self, tmp_path):
        path = tmp_path / "dvx_dataset.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(DatasetSpecError) as excinfo:
            load_spec(str(path))
        assert "顶层必须" in str(excinfo.value)
        assert excinfo.value.hint

    def test_empty_file(self, tmp_path):
        path = tmp_path / "dvx_dataset.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DatasetSpecError):
            load_spec(str(path))

    def test_missing_required_keys(self, tmp_path):
        for key in ("version", "data_type", "layout"):
            payload = _payload(tmp_path)
            payload.pop(key)
            _assert_spec_error(tmp_path, payload, match=key)


# ---------------------------------------------------------------------------
# 失败路径：规则 1（version / data_type / layout 组合）
# ---------------------------------------------------------------------------


class TestIdentityValidation:
    @pytest.mark.parametrize("version", [2, 0, "1", True, None])
    def test_invalid_version(self, tmp_path, version):
        payload = _payload(tmp_path)
        payload["version"] = version
        _assert_spec_error(tmp_path, payload, match="version")

    def test_invalid_data_type(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, data_type="foo"), match="data_type")

    def test_invalid_layout(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, layout="foo"), match="layout")

    def test_classification_layout_mismatch(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, layout="labelme"), match="组合非法")

    def test_detection_layout_mismatch(self, tmp_path):
        payload = _payload(tmp_path, data_type="detection", layout="labelme")
        payload["layout"] = "folder_per_class"
        _assert_spec_error(tmp_path, payload, match="组合非法")

    def test_segmentation_layout_mismatch(self, tmp_path):
        _assert_spec_error(
            tmp_path,
            _payload(tmp_path, data_type="segmentation", layout="folder_per_class"),
            match="组合非法",
        )


# ---------------------------------------------------------------------------
# 失败路径：规则 2/3（classes 与 tasks）
# ---------------------------------------------------------------------------


class TestClassesAndTasksValidation:
    def test_missing_classes(self, tmp_path):
        payload = _payload(tmp_path)
        payload.pop("classes")
        _assert_spec_error(tmp_path, payload, match="classes")

    def test_empty_classes(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, classes=[]), match="不能为空")

    def test_duplicate_classes(self, tmp_path):
        payload = _payload(tmp_path, classes=["cat", "dog", "cat"])
        _assert_spec_error(tmp_path, payload, match="重复类别")

    def test_classes_not_list(self, tmp_path):
        payload = _payload(tmp_path)
        payload["classes"] = "cat"
        _assert_spec_error(tmp_path, payload, match="必须是列表")

    def test_classes_non_string(self, tmp_path):
        payload = _payload(tmp_path)
        payload["classes"] = ["cat", 1]
        _assert_spec_error(tmp_path, payload, match="非空字符串")

    def test_multitask_requires_tasks(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification")
        payload["classes"] = None
        _assert_spec_error(tmp_path, payload, match="至少一个 task")

    def test_multitask_rejects_top_level_classes(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification")
        payload["tasks"] = [{"name": "a", "path": "a", "classes": ["x"]}]
        _assert_spec_error(tmp_path, payload, match="不得同时给出顶层 classes")

    def test_multitask_empty_tasks(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = []
        _assert_spec_error(tmp_path, payload, match="至少一个 task")

    @pytest.mark.parametrize(
        "task, match",
        [
            ({"path": "a", "classes": ["x"]}, "name"),
            ({"name": "a", "classes": ["x"]}, "path"),
            ({"name": "a", "path": "a"}, "classes"),
            ({"name": "a", "path": "a", "classes": []}, "不能为空"),
            ({"name": "a", "path": "a", "classes": ["x", "x"]}, "重复类别"),
            ({"name": "", "path": "a", "classes": ["x"]}, "name"),
        ],
    )
    def test_multitask_bad_task(self, tmp_path, task, match):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = [task]
        _assert_spec_error(tmp_path, payload, match=match)

    def test_multitask_duplicate_task_names(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = [
            {"name": "same", "path": "a", "classes": ["x"]},
            {"name": "same", "path": "b", "classes": ["y"]},
        ]
        _assert_spec_error(tmp_path, payload, match="任务名重复")

    def test_tasks_only_for_multitask(self, tmp_path):
        payload = _payload(tmp_path)
        payload["tasks"] = [{"name": "a", "path": "a", "classes": ["x"]}]
        _assert_spec_error(tmp_path, payload, match="仅用于")

    def test_tasks_not_list(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = {"name": "a"}
        _assert_spec_error(tmp_path, payload, match="tasks 必须是列表")

    def test_same_class_name_allowed_across_tasks(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = [
            {"name": "a", "path": "a", "classes": ["无", "有"]},
            {"name": "b", "path": "b", "classes": ["无", "有"]},
        ]
        spec = _load_payload(tmp_path, payload)
        assert spec.class_names("a") == ["无", "有"]
        assert spec.class_names("b") == ["无", "有"]


# ---------------------------------------------------------------------------
# 失败路径：规则 4（seg_mode / ignore_index）
# ---------------------------------------------------------------------------


class TestSegModeValidation:
    def test_invalid_seg_mode(self, tmp_path):
        payload = _payload(tmp_path, data_type="segmentation", layout="labelme")
        payload["seg_mode"] = "instance"
        _assert_spec_error(tmp_path, payload, match="seg_mode")

    def test_seg_mode_not_allowed_for_detection(self, tmp_path):
        payload = _payload(tmp_path, data_type="detection", layout="labelme")
        payload["seg_mode"] = "query"
        _assert_spec_error(tmp_path, payload, match="仅 segmentation")

    def test_seg_mode_semantic_allowed_for_detection(self, tmp_path):
        payload = _payload(tmp_path, data_type="detection", layout="labelme", seg_mode="semantic")
        spec = _load_payload(tmp_path, payload)
        assert spec.seg_mode == "semantic"

    def test_ignore_index_not_int(self, tmp_path):
        payload = _payload(tmp_path, data_type="segmentation", layout="labelme")
        payload["ignore_index"] = "255"
        _assert_spec_error(tmp_path, payload, match="ignore_index")

    def test_ignore_index_bool(self, tmp_path):
        payload = _payload(tmp_path, data_type="segmentation", layout="labelme")
        payload["ignore_index"] = True
        _assert_spec_error(tmp_path, payload, match="ignore_index")


# ---------------------------------------------------------------------------
# 失败路径：规则 5（splits）
# ---------------------------------------------------------------------------


class TestSplitsValidation:
    def test_invalid_mode(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "random"})
        _assert_spec_error(tmp_path, payload, match="splits.mode")

    def test_ratio_sum_not_one(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "ratio": {"train": 0.7, "val": 0.2}})
        _assert_spec_error(tmp_path, payload, match="之和")

    def test_ratio_unknown_split_name(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={"mode": "ratio", "ratio": {"train": 0.8, "valid": 0.2}},
        )
        _assert_spec_error(tmp_path, payload, match="非法切分名")

    def test_ratio_missing_train(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "ratio": {"val": 1.0}})
        _assert_spec_error(tmp_path, payload, match="train")

    def test_ratio_negative(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={"mode": "ratio", "ratio": {"train": 1.2, "val": -0.2}},
        )
        _assert_spec_error(tmp_path, payload, match="负数")

    def test_ratio_not_number(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={"mode": "ratio", "ratio": {"train": "0.8", "val": "0.2"}},
        )
        _assert_spec_error(tmp_path, payload, match="必须是数字")

    def test_ratio_not_mapping(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "ratio": 0.8})
        _assert_spec_error(tmp_path, payload, match="ratio")

    def test_ratio_empty(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "ratio": {}})
        _assert_spec_error(tmp_path, payload, match="ratio")

    def test_list_without_list_file(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, splits={"mode": "list"}), match="list_file")

    def test_list_file_empty_string(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "list", "list_file": ""})
        _assert_spec_error(tmp_path, payload, match="list_file")

    def test_dir_without_dir_map(self, tmp_path):
        _assert_spec_error(tmp_path, _payload(tmp_path, splits={"mode": "dir"}), match="dir_map")

    def test_dir_empty_dir_map(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "dir", "dir_map": {}})
        _assert_spec_error(tmp_path, payload, match="dir_map")

    def test_dir_map_unknown_split(self, tmp_path):
        payload = _payload(
            tmp_path,
            splits={"mode": "dir", "dir_map": {"train": "t", "valid": "v"}},
        )
        _assert_spec_error(tmp_path, payload, match="非法切分名")

    def test_dir_map_missing_train(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "dir", "dir_map": {"val": "v"}})
        _assert_spec_error(tmp_path, payload, match="train")

    def test_dir_map_value_not_string(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "dir", "dir_map": {"train": 123}})
        _assert_spec_error(tmp_path, payload, match="dir_map")

    def test_seed_not_int(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "seed": "42"})
        _assert_spec_error(tmp_path, payload, match="seed")

    def test_invalid_stratify(self, tmp_path):
        payload = _payload(tmp_path, splits={"mode": "ratio", "stratify": "label"})
        _assert_spec_error(tmp_path, payload, match="stratify")

    def test_splits_not_mapping(self, tmp_path):
        payload = _payload(tmp_path, splits=["ratio"])
        _assert_spec_error(tmp_path, payload, match="splits 必须是映射")


# ---------------------------------------------------------------------------
# 失败路径：规则 6（root）与 parse
# ---------------------------------------------------------------------------


class TestRootAndParseValidation:
    def test_root_missing_directory(self, tmp_path):
        payload = _payload(tmp_path)
        payload["root"] = str(tmp_path / "not-exist")
        _assert_spec_error(tmp_path, payload, match="不存在")

    def test_root_is_file(self, tmp_path):
        file_path = tmp_path / "image.jpg"
        file_path.write_bytes(b"x")
        payload = _payload(tmp_path)
        payload["root"] = str(file_path)
        _assert_spec_error(tmp_path, payload, match="不是目录")

    def test_root_none(self, tmp_path):
        payload = _payload(tmp_path)
        payload["root"] = None
        _assert_spec_error(tmp_path, payload, match="root")

    def test_root_empty_string(self, tmp_path):
        payload = _payload(tmp_path)
        payload["root"] = ""
        _assert_spec_error(tmp_path, payload, match="root")

    @pytest.mark.parametrize(
        "parse, match",
        [
            ("not-a-mapping", "parse 必须是映射"),
            ({"recursive": "yes"}, "recursive"),
            ({"image_ext": []}, "image_ext"),
            ({"image_ext": [".jpg", 3]}, "扩展名"),
            ({"labelme": 3}, "labelme"),
            ({"labelme": {"image_data": "raise"}}, "image_data"),
            ({"labelme": {"ignore_labels": "x"}}, "ignore_labels"),
            ({"labelme": {"ignore_labels": [""]}}, "非空字符串"),
            ({"labelme": {"shape_types": "x"}}, "shape_types"),
            ({"labelme": {"shape_types": {"rectangle": "foo"}}}, "目标任务"),
            ({"labelme": {"shape_types": {"": "detection"}}}, "形状类型"),
        ],
    )
    def test_bad_parse(self, tmp_path, parse, match):
        _assert_spec_error(tmp_path, _payload(tmp_path, parse=parse), match=match)

    def test_unknown_parse_key_only_warns(self, tmp_path):
        payload = _payload(tmp_path, parse={"labelme": {"labelme_ignore": []}})
        with pytest.warns(UserWarning, match="未知字段"):
            spec = _load_payload(tmp_path, payload)
        assert spec.parse == ParseConfig()


# ---------------------------------------------------------------------------
# class_names / num_classes 参数校验
# ---------------------------------------------------------------------------


class TestClassNamesAPI:
    def test_multitask_requires_task(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = [{"name": "a", "path": "a", "classes": ["x"]}]
        spec = _load_payload(tmp_path, payload)
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.class_names()
        assert "必须指定 task" in str(excinfo.value)
        assert excinfo.value.hint

    def test_multitask_unknown_task(self, tmp_path):
        payload = _payload(tmp_path, data_type="multi_task_classification", classes=None)
        payload["tasks"] = [{"name": "a", "path": "a", "classes": ["x"]}]
        spec = _load_payload(tmp_path, payload)
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.num_classes("nope")
        assert "未知任务" in str(excinfo.value)
        assert "a" in (excinfo.value.hint or "")

    def test_single_label_rejects_task(self, tmp_path):
        spec = _load_payload(tmp_path, _payload(tmp_path))
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.class_names("cat")
        assert "不接受 task" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 手工构造 DVXDatasetSpec 的 validate 路径
# ---------------------------------------------------------------------------


def _manual_spec(tmp_path, **overrides):
    """手工构造一个默认合法的检测 spec。"""
    root = tmp_path / "manual-root"
    root.mkdir(exist_ok=True)
    fields = {
        "version": 1,
        "data_type": "detection",
        "layout": "labelme",
        "root": str(root),
        "classes": ["a", "b"],
        "tasks": None,
        "splits": None,
        "parse": ParseConfig(),
    }
    fields.update(overrides)
    return DVXDatasetSpec(**fields)


class TestManualSpec:
    def test_validate_ok(self, tmp_path):
        spec = _manual_spec(tmp_path)
        spec.validate()
        assert spec.class_names() == ["a", "b"]
        assert spec.num_classes() == 2
        assert spec.resolve_root() == str(tmp_path / "manual-root")

    def test_validate_bad_version(self, tmp_path):
        spec = _manual_spec(tmp_path, version=3)
        with pytest.raises(DatasetSpecError):
            spec.validate()

    def test_validate_bad_parse_type(self, tmp_path):
        spec = _manual_spec(tmp_path, parse={"recursive": True})
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.validate()
        assert "ParseConfig" in str(excinfo.value)

    def test_validate_bad_parse_values(self, tmp_path):
        spec = _manual_spec(tmp_path, parse=ParseConfig(image_ext=("jpg",)))
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.validate()
        assert "扩展名" in str(excinfo.value)

    def test_validate_bad_split_config_type(self, tmp_path):
        spec = _manual_spec(tmp_path, splits={"mode": "ratio"})
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.validate()
        assert "SplitConfig" in str(excinfo.value)

    def test_validate_bad_ratio(self, tmp_path):
        spec = _manual_spec(tmp_path, splits=SplitConfig(mode="ratio", ratio={"train": 0.5}))
        with pytest.raises(DatasetSpecError):
            spec.validate()

    def test_validate_bad_dir_map(self, tmp_path):
        spec = _manual_spec(tmp_path, splits=SplitConfig(mode="dir", dir_map=None))
        with pytest.raises(DatasetSpecError):
            spec.validate()

    def test_validate_tasks_only_for_multitask(self, tmp_path):
        tasks = [TaskSpec(name="a", path="a", classes=["x"])]
        spec = _manual_spec(tmp_path, tasks=tasks)
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.validate()
        assert "仅用于" in str(excinfo.value)

    def test_validate_bad_task(self, tmp_path):
        tasks = [TaskSpec(name="", path="a", classes=["x"])]
        spec = _manual_spec(
            tmp_path,
            data_type="multi_task_classification",
            layout="folder_per_class",
            classes=None,
            tasks=tasks,
        )
        with pytest.raises(DatasetSpecError) as excinfo:
            spec.validate()
        assert "name" in str(excinfo.value)

    def test_resolve_root_relative_without_spec_path(self, tmp_path, monkeypatch):
        (tmp_path / "rel-data").mkdir()
        monkeypatch.chdir(tmp_path)
        spec = _manual_spec(tmp_path, root="rel-data")
        assert spec.resolve_root() == os.path.realpath(str(tmp_path / "rel-data"))
