# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 标准数据集规格（dvx_dataset.yaml）的 schema 定义与校验。
"""DVX 标准数据集规格（``dvx_dataset.yaml``）的 schema 定义与校验。

设计原则（契约 P1-0）：

- 一个数据集 = 一个根目录 + 递归扫描，多级目录只影响路径、不携带语义；
- 数据规格只描述"有什么标注"，训练目标由训练配置决定（P2 再接线）；
- 规格中的路径在内存里统一为绝对路径，展示时才转成相对路径。

本模块自身只依赖 pyyaml 与 ``dvisionix.errors``，不导入 torch / cv2 / numpy。

用法::

    from dvisionix.data.spec.schema import load_spec

    spec = load_spec("project/data/cls/dvx_dataset.yaml")
    print(spec.class_names(), spec.num_classes(), spec.resolve_root())

校验失败一律抛 :class:`dvisionix.errors.DatasetSpecError`，消息包含"发生了什么 +
应该怎么办"（hint）。
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml

from dvisionix.errors import DatasetSpecError

__all__ = [
    "CLASSIFICATION_TYPES",
    "DATA_TYPES",
    "DATA_TYPE_LAYOUTS",
    "LABELME_IMAGE_DATA_MODES",
    "LABELME_SHAPE_TARGETS",
    "LAYOUTS",
    "RATIO_SUM_TOLERANCE",
    "SEG_MODES",
    "SPLIT_MODES",
    "SPLIT_NAMES",
    "DVXDatasetSpec",
    "ParseConfig",
    "SplitConfig",
    "TaskSpec",
    "load_spec",
]

#: 支持的数据类型
DATA_TYPES = ("classification", "multi_task_classification", "detection", "segmentation")
#: 支持的目录布局
LAYOUTS = ("folder_per_class", "labelme")
#: data_type 与 layout 的合法组合（契约 P1-2 校验规则 1）
DATA_TYPE_LAYOUTS: dict[str, tuple[str, ...]] = {
    "classification": ("folder_per_class",),
    "multi_task_classification": ("folder_per_class",),
    "detection": ("labelme",),
    "segmentation": ("labelme",),
}
#: 单标签 / 多任务分类（决定 stratify 默认值）
CLASSIFICATION_TYPES = ("classification", "multi_task_classification")
#: 切分模式
SPLIT_MODES = ("ratio", "list", "dir")
#: 合法的 split 名字
SPLIT_NAMES = ("train", "val", "test")
#: 分割任务模式（本期只实现 semantic）
SEG_MODES = ("semantic", "query")
#: labelme 内嵌图像数据的处理方式
LABELME_IMAGE_DATA_MODES = ("ignore", "error")
#: labelme 形状类型 -> 目标任务
LABELME_SHAPE_TARGETS = ("detection", "segmentation")
#: ratio 之和的容差
RATIO_SUM_TOLERANCE = 1e-6

#: labelme 形状类型的默认语义（契约 P1-2）
DEFAULT_LABELME_SHAPE_TYPES: dict[str, str] = {
    "rectangle": "detection",
    "polygon": "segmentation",
}

_SPEC_KEYS = {
    "version",
    "data_type",
    "layout",
    "root",
    "classes",
    "tasks",
    "splits",
    "parse",
    "seg_mode",
    "ignore_index",
}
_TASK_KEYS = {"name", "path", "classes"}
_SPLIT_KEYS = {"mode", "ratio", "seed", "stratify", "list_file", "dir_map"}
_PARSE_KEYS = {"recursive", "image_ext", "labelme"}
_LABELME_KEYS = {"ignore_labels", "image_data", "shape_types"}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """多任务分类中的一个子任务。

    Args:
        name: 任务名（在 spec 内唯一），如 ``licence``。
        path: 相对数据集根目录的子目录，如 ``T15C_1CCD6_licence``。
        classes: 该任务独立的类别表（顺序即 label id）。
    """

    name: str
    path: str
    classes: list[str]


@dataclass
class SplitConfig:
    """切分配置（契约 P1-2 / P1-5）。

    Args:
        mode: ``ratio`` / ``list`` / ``dir``。
        ratio: mode=ratio 时的切分比例，键 ⊆ {train, val, test}，和为 1。
        seed: mode=ratio 时的随机种子。
        stratify: ``class`` / ``none`` / ``None``；``None`` 表示不启用分层。
        list_file: mode=list 时必填，JSON/YAML 相对路径清单。
        dir_map: mode=dir 时必填，``{split: 相对目录}``。
    """

    mode: str = "ratio"
    ratio: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
    seed: int = 42
    stratify: str | None = None
    list_file: str | None = None
    dir_map: dict[str, str] | None = None


@dataclass
class ParseConfig:
    """扫描/解析标注的行为配置（契约 P1-2）。

    Args:
        recursive: 是否递归扫描子目录。
        image_ext: 参与扫描的图片扩展名（小写、带前导点）。
        labelme_ignore_labels: labelme 中直接丢弃的标签名。
        labelme_image_data: labelme 内嵌 ``imageData`` 的处理方式（ignore / error）。
        labelme_shape_types: labelme 形状类型 -> 目标任务（detection / segmentation）。
    """

    recursive: bool = True
    image_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    labelme_ignore_labels: tuple[str, ...] = ("__ignore__", "__background__")
    labelme_image_data: str = "ignore"
    labelme_shape_types: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LABELME_SHAPE_TYPES)
    )


@dataclass
class DVXDatasetSpec:
    """一个 DVX 标准数据集的完整规格。

    Args:
        version: 规格版本，当前只支持 1。
        data_type: classification / multi_task_classification / detection / segmentation。
        layout: folder_per_class / labelme。
        root: 数据集根目录；``from_yaml`` 之后为绝对路径。
        classes: 单标签空间的类别表（多任务时为 None）。
        tasks: 多任务分类的子任务列表（其余类型为 None）。
        splits: 切分配置；允许为 None（表示未配置切分）。
        parse: 扫描/解析配置。
        seg_mode: 分割模式，仅 segmentation 可设。
        ignore_index: 分割忽略像素值。
        spec_path: yaml 绝对路径（手工构造时可为空）。
    """

    version: int
    data_type: str
    layout: str
    root: str
    classes: list[str] | None
    tasks: list[TaskSpec] | None
    splits: SplitConfig | None
    parse: ParseConfig
    seg_mode: str = "semantic"
    ignore_index: int = 255
    spec_path: str = ""

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "DVXDatasetSpec":
        """读取并校验 spec 文件，返回 ``DVXDatasetSpec``。

        Args:
            path: yaml 文件路径（相对路径按当前工作目录解析）。

        Returns:
            校验通过的 spec；``root`` / ``spec_path`` 均为绝对路径。

        Raises:
            DatasetSpecError: 文件不存在、YAML 语法错误、结构/取值不合法。
        """
        spec_path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(spec_path):
            raise DatasetSpecError(
                f"spec 文件不存在: {spec_path}",
                hint="请确认 --spec 指向实际存在的 dvx_dataset.yaml",
            )
        try:
            with open(spec_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except OSError as exc:
            raise DatasetSpecError(
                f"无法读取 spec 文件: {spec_path}（{exc}）",
                hint="请检查文件权限与编码（要求 UTF-8）",
            ) from exc
        except yaml.YAMLError as exc:
            raise DatasetSpecError(
                f"spec YAML 解析失败: {spec_path}（{exc}）",
                hint="请检查缩进、冒号与引号是否合法（中文建议加引号）",
            ) from exc

        if not isinstance(raw, Mapping):
            raise DatasetSpecError(
                f"spec 顶层必须是映射（mapping）: {spec_path}" f"（实际为 {type(raw).__name__}）",
                hint="参考契约 P1-2 的 yaml 结构，顶层应包含 version/data_type/layout/root 等键",
            )
        mapping = dict(raw)
        _warn_unknown_keys(mapping, _SPEC_KEYS, f"spec {os.path.basename(spec_path)}")

        version = _require_key(mapping, "version", "spec")
        data_type = _require_key(mapping, "data_type", "spec")
        layout = _require_key(mapping, "layout", "spec")
        root = _ensure_non_empty_str(mapping.get("root", "."), "root")
        classes = _optional_str_list(mapping.get("classes"), "classes")
        tasks = _parse_tasks(mapping.get("tasks"))
        splits = _parse_splits(mapping.get("splits"), data_type, spec_path)
        parse = _parse_parse_config(mapping.get("parse"))

        spec = cls(
            version=version,
            data_type=data_type,
            layout=layout,
            root=root,
            classes=classes,
            tasks=tasks,
            splits=splits,
            parse=parse,
            seg_mode=mapping.get("seg_mode", "semantic"),
            ignore_index=mapping.get("ignore_index", 255),
            spec_path=spec_path,
        )
        spec.validate()
        # 内存中统一存绝对路径（validate 已确认可解析且存在）
        spec.root = spec.resolve_root()
        return spec

    # -- 公共接口 -----------------------------------------------------------

    def validate(self) -> None:
        """校验 spec 的全部约束；失败抛 ``DatasetSpecError``。"""
        self._validate_identity()
        self._validate_classes_and_tasks()
        self._validate_seg_mode()
        self._validate_parse()
        self._validate_splits()
        self._validate_root()

    def class_names(self, task: str | None = None) -> list[str]:
        """返回类别表（副本）。

        Args:
            task: 多任务分类必填的任务名；单标签数据集必须为 None。

        Raises:
            DatasetSpecError: 多任务未指定/指定了未知任务，或单标签误传 task。
        """
        if self.data_type == "multi_task_classification":
            names = [item.name for item in (self.tasks or [])]
            if task is None:
                raise DatasetSpecError(
                    "多任务分类数据集必须指定 task",
                    hint=f"可选任务: {names}",
                )
            for item in self.tasks or []:
                if item.name == task:
                    return list(item.classes)
            raise DatasetSpecError(
                f"未知任务: {task!r}",
                hint=f"可选任务: {names}",
            )
        if task is not None:
            raise DatasetSpecError(
                f"单标签数据集不接受 task 参数（收到 {task!r}）",
                hint="请传入 task=None，或改用多任务分类数据集",
            )
        return list(self.classes or [])

    def num_classes(self, task: str | None = None) -> int:
        """返回类别数（多任务需指定 task）。"""
        return len(self.class_names(task))

    def resolve_root(self) -> str:
        """返回数据集根目录的绝对路径（相对路径以 spec 文件所在目录为基准）。"""
        root = self.root if isinstance(self.root, str) else ""
        if os.path.isabs(root):
            return os.path.normpath(root)
        if self.spec_path:
            base = os.path.dirname(os.path.abspath(self.spec_path))
        else:
            base = os.getcwd()
        return os.path.normpath(os.path.abspath(os.path.join(base, root or ".")))

    # -- 校验实现 -----------------------------------------------------------

    def _validate_identity(self) -> None:
        """校验 version / data_type / layout 及其组合。"""
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise DatasetSpecError(
                f"version 必须是整数 1（当前为 {self.version!r}）",
                hint="请在 dvx_dataset.yaml 顶部写 version: 1",
            )
        if self.version != 1:
            raise DatasetSpecError(
                f"不支持的 spec version: {self.version!r}",
                hint="当前仅支持 version: 1，请升级实现或改用旧版格式",
            )
        if self.data_type not in DATA_TYPES:
            raise DatasetSpecError(
                f"非法的 data_type: {self.data_type!r}",
                hint=f"可选值: {list(DATA_TYPES)}",
            )
        if self.layout not in LAYOUTS:
            raise DatasetSpecError(
                f"非法的 layout: {self.layout!r}",
                hint=f"可选值: {list(LAYOUTS)}",
            )
        allowed_layouts = DATA_TYPE_LAYOUTS[self.data_type]
        if self.layout not in allowed_layouts:
            raise DatasetSpecError(
                f"data_type={self.data_type!r} 与 layout={self.layout!r} 组合非法",
                hint=f"{self.data_type} 只支持 layout: {list(allowed_layouts)}",
            )

    def _validate_classes_and_tasks(self) -> None:
        """校验顶层 classes 与 tasks 的互斥性、非空性与唯一性。"""
        if self.data_type == "multi_task_classification":
            if self.classes is not None:
                raise DatasetSpecError(
                    "multi_task_classification 不得同时给出顶层 classes",
                    hint="请把类别写在 tasks[*].classes 中（每个任务独立类表）",
                )
            if not self.tasks:
                raise DatasetSpecError(
                    "multi_task_classification 必须提供至少一个 task",
                    hint="请在 yaml 中增加 tasks: [{name, path, classes}, ...]",
                )
            seen_names: set[str] = set()
            for index, task in enumerate(self.tasks):
                if not isinstance(task, TaskSpec):
                    raise DatasetSpecError(
                        f"tasks[{index}] 必须是 TaskSpec（当前为 {type(task).__name__}）",
                        hint="请用 DVXDatasetSpec.from_yaml 解析，或手工构造 TaskSpec",
                    )
                name = _ensure_non_empty_str(task.name, f"tasks[{index}].name")
                if name in seen_names:
                    raise DatasetSpecError(
                        f"任务名重复: {name!r}",
                        hint="多任务分类的 task.name 必须唯一，样本按 name 归属任务",
                    )
                seen_names.add(name)
                _ensure_non_empty_str(task.path, f"tasks[{index}].path")
                task_classes = _ensure_str_list(task.classes, f"tasks[{index}].classes")
                if not task_classes:
                    raise DatasetSpecError(
                        f"任务 {name!r} 的 classes 不能为空",
                        hint="每个任务至少要有 1 个类别",
                    )
                _ensure_unique(task_classes, f"任务 {name!r} 的 classes")
            return

        if self.tasks:
            raise DatasetSpecError(
                "tasks 仅用于 multi_task_classification",
                hint=f"当前 data_type={self.data_type!r}，请改用顶层 classes",
            )
        if self.classes is None:
            raise DatasetSpecError(
                f"{self.data_type} 数据集必须提供 classes",
                hint="请在 yaml 中增加 classes: [类名, ...]（顺序即 label id）",
            )
        classes = _ensure_str_list(self.classes, "classes")
        if not classes:
            raise DatasetSpecError(
                "classes 不能为空",
                hint="至少声明 1 个类别；顺序即 label id，后续不可随意调整",
            )
        _ensure_unique(classes, "classes")

    def _validate_seg_mode(self) -> None:
        """校验 seg_mode 枚举与适用范围，以及 ignore_index 类型。"""
        if self.seg_mode not in SEG_MODES:
            raise DatasetSpecError(
                f"非法的 seg_mode: {self.seg_mode!r}",
                hint=f"可选值: {list(SEG_MODES)}（本期只实现 semantic）",
            )
        if self.data_type != "segmentation" and self.seg_mode != "semantic":
            raise DatasetSpecError(
                f"seg_mode 仅 segmentation 数据集可设置（当前 data_type={self.data_type!r}）",
                hint="请删除 seg_mode 字段，或把它改回默认的 semantic",
            )
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise DatasetSpecError(
                f"ignore_index 必须是整数（当前为 {self.ignore_index!r}）",
                hint="语义分割常用 255 或 -1，需与训练配置保持一致",
            )

    def _validate_parse(self) -> None:
        """校验 parse（含 labelme 子项）的取值。"""
        parse = self.parse
        if not isinstance(parse, ParseConfig):
            raise DatasetSpecError(
                f"parse 必须是 ParseConfig（当前为 {type(parse).__name__}）",
                hint="请用 DVXDatasetSpec.from_yaml 解析，或手工构造 ParseConfig",
            )
        if not isinstance(parse.recursive, bool):
            raise DatasetSpecError(
                f"parse.recursive 必须是布尔值（当前为 {parse.recursive!r}）",
                hint="写 true / false（不要加引号）",
            )
        if not _is_non_empty_sequence(parse.image_ext):
            raise DatasetSpecError(
                "parse.image_ext 必须是非空列表",
                hint="例如 [.jpg, .jpeg, .png, .bmp]",
            )
        for ext in parse.image_ext:
            if not isinstance(ext, str) or not ext.startswith(".") or len(ext) < 2:
                raise DatasetSpecError(
                    f"非法的图片扩展名: {ext!r}",
                    hint="扩展名必须带前导点，例如 .jpg",
                )
        if not _is_sequence(parse.labelme_ignore_labels):
            raise DatasetSpecError(
                "parse.labelme.ignore_labels 必须是列表",
                hint="例如 [__ignore__, __background__]",
            )
        for label in parse.labelme_ignore_labels:
            if not isinstance(label, str) or not label:
                raise DatasetSpecError(
                    f"parse.labelme.ignore_labels 中的标签必须是非空字符串（当前为 {label!r}）",
                    hint="直接写标签名字符串即可",
                )
        if parse.labelme_image_data not in LABELME_IMAGE_DATA_MODES:
            raise DatasetSpecError(
                f"非法的 parse.labelme.image_data: {parse.labelme_image_data!r}",
                hint=f"可选值: {list(LABELME_IMAGE_DATA_MODES)}",
            )
        if not isinstance(parse.labelme_shape_types, Mapping):
            raise DatasetSpecError(
                "parse.labelme.shape_types 必须是映射（形状类型 -> 目标任务）",
                hint="例如 {rectangle: detection, polygon: segmentation}",
            )
        for shape, target in parse.labelme_shape_types.items():
            if not isinstance(shape, str) or not shape:
                raise DatasetSpecError(
                    f"parse.labelme.shape_types 的形状类型必须是非空字符串（当前为 {shape!r}）",
                    hint="例如 rectangle / polygon",
                )
            if target not in LABELME_SHAPE_TARGETS:
                raise DatasetSpecError(
                    f"parse.labelme.shape_types[{shape!r}] 的目标任务非法: {target!r}",
                    hint=f"可选值: {list(LABELME_SHAPE_TARGETS)}",
                )

    def _validate_splits(self) -> None:
        """校验切分配置（mode / ratio / stratify / list_file / dir_map）。"""
        config = self.splits
        if config is None:
            return
        if not isinstance(config, SplitConfig):
            raise DatasetSpecError(
                f"splits 必须是 SplitConfig（当前为 {type(config).__name__}）",
                hint="请用 DVXDatasetSpec.from_yaml 解析，或手工构造 SplitConfig",
            )
        if config.mode not in SPLIT_MODES:
            raise DatasetSpecError(
                f"非法的 splits.mode: {config.mode!r}",
                hint=f"可选值: {list(SPLIT_MODES)}",
            )
        if isinstance(config.seed, bool) or not isinstance(config.seed, int):
            raise DatasetSpecError(
                f"splits.seed 必须是整数（当前为 {config.seed!r}）",
                hint="保证切分可复现，例如 seed: 42",
            )
        if config.stratify not in (None, "class", "none"):
            raise DatasetSpecError(
                f"非法的 splits.stratify: {config.stratify!r}",
                hint="可选值: class / none（不写则由 data_type 决定默认值）",
            )
        if config.mode == "ratio":
            self._validate_ratio(config.ratio)
        elif config.mode == "list":
            _ensure_non_empty_str(config.list_file, "splits.list_file")
        else:  # dir
            self._validate_dir_map(config.dir_map)

    @staticmethod
    def _validate_ratio(ratio: Any) -> None:
        """校验 ratio 键集合、数值与总和。"""
        if not isinstance(ratio, Mapping) or not ratio:
            raise DatasetSpecError(
                "splits.ratio 必须是非空映射",
                hint="例如 {train: 0.8, val: 0.2}",
            )
        unknown = [key for key in ratio if key not in SPLIT_NAMES]
        if unknown:
            raise DatasetSpecError(
                f"splits.ratio 含非法切分名: {unknown}",
                hint=f"只允许 {list(SPLIT_NAMES)}",
            )
        if "train" not in ratio:
            raise DatasetSpecError(
                "splits.ratio 缺少 train",
                hint="切分必须包含 train（例如 train: 0.8, val: 0.2）",
            )
        total = 0.0
        for name, value in ratio.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DatasetSpecError(
                    f"splits.ratio[{name!r}] 必须是数字（当前为 {value!r}）",
                    hint="例如 train: 0.8",
                )
            if value < 0:
                raise DatasetSpecError(
                    f"splits.ratio[{name!r}] 不能为负数（当前为 {value!r}）",
                    hint="比例应为 0~1 之间的数",
                )
            total += float(value)
        if abs(total - 1.0) > RATIO_SUM_TOLERANCE:
            raise DatasetSpecError(
                f"splits.ratio 之和必须为 1.0（当前为 {total:.9f}）",
                hint=f"容差 {RATIO_SUM_TOLERANCE}，例如 train: 0.8, val: 0.2",
            )

    @staticmethod
    def _validate_dir_map(dir_map: Any) -> None:
        """校验 dir 模式的目录映射。"""
        if not isinstance(dir_map, Mapping) or not dir_map:
            raise DatasetSpecError(
                "mode=dir 时必须提供非空的 splits.dir_map",
                hint="例如 dir_map: {train: train_data/train, val: train_data/val}",
            )
        unknown = [key for key in dir_map if key not in SPLIT_NAMES]
        if unknown:
            raise DatasetSpecError(
                f"splits.dir_map 含非法切分名: {unknown}",
                hint=f"只允许 {list(SPLIT_NAMES)}",
            )
        if "train" not in dir_map:
            raise DatasetSpecError(
                "splits.dir_map 缺少 train",
                hint="切分必须包含 train 目录",
            )
        for name, subdir in dir_map.items():
            if not isinstance(subdir, str) or not subdir.strip():
                raise DatasetSpecError(
                    f"splits.dir_map[{name!r}] 必须是相对目录字符串（当前为 {subdir!r}）",
                    hint="例如 train: train_data/train（相对 spec.root 或 task.path）",
                )

    def _validate_root(self) -> None:
        """校验 root 可解析、存在且是目录。"""
        root_value = self.root
        if not isinstance(root_value, str) or not root_value.strip():
            raise DatasetSpecError(
                f"root 必须是非空字符串（当前为 {root_value!r}）",
                hint="root 相对 spec 文件所在目录，例如 root: .",
            )
        resolved = self.resolve_root()
        if not os.path.exists(resolved):
            raise DatasetSpecError(
                f"数据集根目录不存在: {resolved}",
                hint="检查 root 是否相对 spec 文件所在目录，或改用绝对路径",
            )
        if not os.path.isdir(resolved):
            raise DatasetSpecError(
                f"数据集根目录不是目录: {resolved}",
                hint="root 必须指向目录，而不是某张图片/标注文件",
            )


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def load_spec(path: str) -> DVXDatasetSpec:
    """读取并校验 ``dvx_dataset.yaml``（契约 P1-2）。

    Args:
        path: spec 文件路径。

    Returns:
        校验通过的 :class:`DVXDatasetSpec`。

    Raises:
        DatasetSpecError: 文件不存在 / YAML 语法错误 / 任何校验失败。
    """
    return DVXDatasetSpec.from_yaml(path)


# ---------------------------------------------------------------------------
# yaml -> dataclass 的解析辅助
# ---------------------------------------------------------------------------


def _parse_tasks(raw: Any) -> list[TaskSpec] | None:
    """把 yaml 中的 tasks 列表转成 TaskSpec 列表（字段缺失交给 validate 报错）。"""
    if raw is None:
        return None
    if not _is_sequence(raw):
        raise DatasetSpecError(
            f"tasks 必须是列表（当前为 {type(raw).__name__}）",
            hint="例如 tasks: [{name: licence, path: T15C_1CCD6_licence, classes: [无, 短牌照]}]",
        )
    tasks: list[TaskSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise DatasetSpecError(
                f"tasks[{index}] 必须是映射（当前为 {type(item).__name__}）",
                hint="每项需包含 name / path / classes",
            )
        mapping = dict(item)
        _warn_unknown_keys(mapping, _TASK_KEYS, f"tasks[{index}]")
        tasks.append(
            TaskSpec(
                name=mapping.get("name"),
                path=mapping.get("path"),
                classes=mapping.get("classes"),
            )
        )
    return tasks


def _parse_splits(raw: Any, data_type: str, spec_path: str) -> SplitConfig | None:
    """把 yaml 中的 splits 段转成 SplitConfig。"""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DatasetSpecError(
            f"splits 必须是映射（当前为 {type(raw).__name__}）",
            hint="例如 splits: {mode: ratio, ratio: {train: 0.8, val: 0.2}}",
        )
    mapping = dict(raw)
    _warn_unknown_keys(mapping, _SPLIT_KEYS, "splits")

    mode = mapping.get("mode", "ratio")
    ratio_raw = mapping.get("ratio")
    if ratio_raw is None:
        ratio: Any = {"train": 0.8, "val": 0.2}
    elif isinstance(ratio_raw, Mapping):
        ratio = {
            key: float(value) if _is_number(value) else value for key, value in ratio_raw.items()
        }
    else:
        ratio = ratio_raw
    stratify = mapping.get("stratify")
    if stratify is None:
        # 契约 P1-2 注释：默认 class（分类）/ none（检测、分割）
        stratify = "class" if data_type in CLASSIFICATION_TYPES else "none"
    dir_map_raw = mapping.get("dir_map")
    dir_map = dict(dir_map_raw) if isinstance(dir_map_raw, Mapping) else dir_map_raw
    list_file_raw = mapping.get("list_file")
    if isinstance(list_file_raw, str) and list_file_raw and not os.path.isabs(list_file_raw):
        # 与 root 一致：相对 spec 文件所在目录解析为绝对路径
        list_file = os.path.normpath(
            os.path.abspath(os.path.join(os.path.dirname(spec_path), list_file_raw))
        )
    else:
        list_file = list_file_raw
    return SplitConfig(
        mode=mode,
        ratio=ratio,
        seed=mapping.get("seed", 42),
        stratify=stratify,
        list_file=list_file,
        dir_map=dir_map,
    )


def _parse_parse_config(raw: Any) -> ParseConfig:
    """把 yaml 中的 parse 段（含 labelme 子段）转成 ParseConfig。"""
    if raw is None:
        return ParseConfig()
    if not isinstance(raw, Mapping):
        raise DatasetSpecError(
            f"parse 必须是映射（当前为 {type(raw).__name__}）",
            hint="例如 parse: {recursive: true, image_ext: [.jpg, .png]}",
        )
    mapping = dict(raw)
    _warn_unknown_keys(mapping, _PARSE_KEYS, "parse")

    defaults = ParseConfig()
    recursive = mapping.get("recursive", defaults.recursive)
    image_ext_raw = mapping.get("image_ext")
    image_ext = (
        tuple(defaults.image_ext) if image_ext_raw is None else _normalize_image_ext(image_ext_raw)
    )

    labelme_raw = mapping.get("labelme")
    ignore_labels: Any = defaults.labelme_ignore_labels
    image_data: Any = defaults.labelme_image_data
    shape_types: Any = dict(defaults.labelme_shape_types)
    if labelme_raw is not None:
        if not isinstance(labelme_raw, Mapping):
            raise DatasetSpecError(
                f"parse.labelme 必须是映射（当前为 {type(labelme_raw).__name__}）",
                hint="例如 labelme: {ignore_labels: [__ignore__], image_data: ignore}",
            )
        labelme = dict(labelme_raw)
        _warn_unknown_keys(labelme, _LABELME_KEYS, "parse.labelme")
        if "ignore_labels" in labelme:
            ignore_labels = _optional_str_tuple(
                labelme["ignore_labels"], "parse.labelme.ignore_labels"
            )
        if "image_data" in labelme:
            image_data = labelme["image_data"]
        if "shape_types" in labelme:
            shape_types_raw = labelme["shape_types"]
            if not isinstance(shape_types_raw, Mapping):
                raise DatasetSpecError(
                    "parse.labelme.shape_types 必须是映射",
                    hint="例如 {rectangle: detection, polygon: segmentation}",
                )
            shape_types = dict(shape_types_raw)

    return ParseConfig(
        recursive=recursive,
        image_ext=image_ext,
        labelme_ignore_labels=ignore_labels,
        labelme_image_data=image_data,
        labelme_shape_types=shape_types,
    )


def _normalize_image_ext(raw: Any) -> tuple[str, ...]:
    """校验并规范化 image_ext（小写、补前导点）。"""
    if not _is_non_empty_sequence(raw):
        raise DatasetSpecError(
            f"parse.image_ext 必须是非空列表（当前为 {raw!r}）",
            hint="例如 [.jpg, .jpeg, .png, .bmp]",
        )
    result: list[str] = []
    missing_dot: list[str] = []
    for ext in raw:
        if not isinstance(ext, str) or not ext.strip():
            raise DatasetSpecError(
                f"parse.image_ext 中的扩展名必须是非空字符串（当前为 {ext!r}）",
                hint="例如 .jpg",
            )
        value = ext.strip().lower()
        if not value.startswith("."):
            missing_dot.append(value)
            value = f".{value}"
        result.append(value)
    if missing_dot:
        warnings.warn(
            f"parse.image_ext 缺少前导点，已自动补全: {missing_dot} -> {[f'.{v}' for v in missing_dot]}",
            UserWarning,
            stacklevel=2,
        )
    return tuple(result)


def _optional_str_list(raw: Any, where: str) -> list[str] | None:
    """按需做"必须是列表"的宽松检查，值类型留给 validate 精确报错。"""
    if raw is None:
        return None
    if not _is_sequence(raw):
        raise DatasetSpecError(
            f"{where} 必须是列表（当前为 {type(raw).__name__}）",
            hint=f"例如 {where}: [cat, dog]",
        )
    return list(raw)


def _optional_str_tuple(raw: Any, where: str) -> tuple[str, ...]:
    """把 yaml 列表转成字符串元组并校验元素。"""
    if not _is_sequence(raw):
        raise DatasetSpecError(
            f"{where} 必须是列表（当前为 {type(raw).__name__}）",
            hint="例如 [__ignore__, __background__]",
        )
    for item in raw:
        if not isinstance(item, str) or not item:
            raise DatasetSpecError(
                f"{where} 中的标签必须是非空字符串（当前为 {item!r}）",
                hint="直接写标签名字符串即可",
            )
    return tuple(raw)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _require_key(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    """取必填键；缺失时抛 DatasetSpecError。"""
    if key not in mapping:
        raise DatasetSpecError(
            f"{where} 缺少必填字段 {key!r}",
            hint=f"请在 dvx_dataset.yaml 中补上 {key}",
        )
    return mapping[key]


def _ensure_non_empty_str(value: Any, where: str) -> str:
    """要求非空字符串并原样返回。"""
    if not isinstance(value, str) or not value.strip():
        raise DatasetSpecError(
            f"{where} 必须是非空字符串（当前为 {value!r}）",
            hint=f"请检查 {where} 的取值",
        )
    return value


def _ensure_str_list(value: Any, where: str) -> list[str]:
    """要求非空字符串列表并返回副本。"""
    if not _is_sequence(value):
        raise DatasetSpecError(
            f"{where} 必须是列表（当前为 {type(value).__name__}）",
            hint=f"例如 {where}: [cat, dog]",
        )
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetSpecError(
                f"{where} 中的类别必须是非空字符串（当前为 {item!r}）",
                hint="类别名按原样使用；若含特殊字符请加引号",
            )
    return list(value)


def _ensure_unique(items: Sequence[str], where: str) -> None:
    """拒绝重复类别名（顺序即 label id，重名会导致映射歧义）。"""
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise DatasetSpecError(
            f"{where} 存在重复类别: {duplicates[:5]}",
            hint="类别顺序即 label id，名称必须唯一",
        )


def _is_number(value: Any) -> bool:
    """判断是否为 int/float（排除 bool）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sequence(value: Any) -> bool:
    """判断是否为 list/tuple（排除 str/bytes）。"""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_non_empty_sequence(value: Any) -> bool:
    """判断是否为非空 list/tuple。"""
    return _is_sequence(value) and len(value) > 0


def _warn_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    """对未知字段发告警（不中断解析，避免"配了没生效"静默失败）。"""
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        warnings.warn(
            f"{where} 含未知字段，已忽略: {unknown}；可用字段: {sorted(allowed)}",
            UserWarning,
            stacklevel=2,
        )
