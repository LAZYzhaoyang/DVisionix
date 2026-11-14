# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集扫描：SampleRef 定义与 scan_samples 调度（契约 P1-4）。
"""DVX 数据集扫描：``SampleRef`` 与 ``scan_samples``（契约 P1-4）。

``scan_samples`` 是"数据集 -> 样本列表"的唯一入口：按 ``spec.layout`` 分派到具体扫描器，
与切分（``dvisionix.data.spec.split``）无关——切分只消费这里的输出，绝不移动原始文件。

分派规则：

- ``layout == "folder_per_class"``：调用
  :func:`dvisionix.data.parsers.folder.scan_folder_per_class`；
  ``multi_task_classification`` 时按 ``spec.tasks`` 逐任务扫描，样本带 ``task=name``。
- ``layout == "labelme"``：构造类别表后调用
  :func:`dvisionix.data.parsers.labelme.scan_labelme`。

``splits.mode == "dir"`` 时（只可能是 folder_per_class），如果 ``dir_map`` 的各目录共享
同一个一级子目录（例如 ``train_data/train`` 与 ``train_data/val`` 共享 ``train_data``），
扫描会自动收敛到这个公共子目录：否则同一批图片会同时以"扁平类目录"和"切片目录"两种
``rel_path`` 被计入，既重复又会让 dir 切分报告"未被覆盖"。这是**自动推断**，因此会向
stderr 打印一行说明（契约 P1-0 第 5 条）；调用方也可以显式传 ``subdir`` 覆盖它。

``scan_samples`` 返回的列表按 ``rel_path`` 升序且 ``rel_path`` 唯一（确定性，契约 P1-4）。

用法::

    from dvisionix.data.spec.schema import load_spec
    from dvisionix.data.spec.scan import scan_samples

    spec = load_spec("project/data/cls/dvx_dataset.yaml")
    for ref in scan_samples(spec):
        print(ref.rel_path, ref.label)
"""

from __future__ import annotations

import os
import posixpath
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dvisionix.data.spec.classes import ClassTable
from dvisionix.data.spec.schema import DVXDatasetSpec
from dvisionix.errors import AnnotationError, DatasetSpecError

if TYPE_CHECKING:
    import numpy as np

__all__ = ["SampleRef", "scan_samples"]


@dataclass
class SampleRef:
    """扫描得到的一个样本：一张图片 + 它的标注（契约 P1-4）。

    图片与标注的路径都是**绝对路径**；``rel_path`` 是相对 ``spec.root`` 的 posix 风格
    路径，是切分、清单匹配与统计的唯一稳定标识。

    Args:
        image_path: 图片绝对路径。
        rel_path: 相对 ``spec.root`` 的 posix 路径（例如 ``cat/x.jpg``）。
        task: 多任务分类的任务名；单任务为 None。
        label: 分类标签（类别 id）；检测/分割为 None。
        boxes: 检测框 ``(N, 4)`` ``float32``，``[x1, y1, x2, y2]`` 绝对像素。
        labels: 检测框类别 ``(N,)`` ``int64``。
        segments: 分割多边形列表，每个 ``(P, 2)`` ``float32`` 顶点数组。
        segment_labels: 分割多边形类别 ``(M,)`` ``int64``，与 ``segments`` 一一对应。
        ann_path: labelme JSON 绝对路径；folder_per_class 为 None。
        image_size: ``(H, W)``，来自 labelme 的 ``imageHeight`` / ``imageWidth``。
        meta: 扫描器附加的调试/统计信息（如 ``class_name`` / ``skipped_shapes``）。
    """

    image_path: str
    rel_path: str
    task: str | None = None
    label: int | None = None
    boxes: np.ndarray | None = None
    labels: np.ndarray | None = None
    segments: list[np.ndarray] | None = None
    segment_labels: np.ndarray | None = None
    ann_path: str | None = None
    image_size: tuple[int, int] | None = None
    meta: dict = field(default_factory=dict)


def scan_samples(spec: DVXDatasetSpec, subdir: str | None = None) -> list[SampleRef]:
    """按 spec 扫描全部样本（与 split 无关），返回按 ``rel_path`` 升序的列表。

    Args:
        spec: 数据集规格；函数开头会调用 ``spec.validate()`` 做一次完整校验。
        subdir: 可选的额外子目录层（仅 folder_per_class 生效），相对
            ``<spec.root>/<task.path>``。默认 None 时按 ``splits.dir_map`` 自动推断
            （见模块 docstring）；显式传入时以传入值为准。

    Returns:
        样本列表，``rel_path`` 相对 ``spec.root`` 且全局唯一，按 ``rel_path`` 升序。

    Raises:
        DatasetSpecError: spec 不合法，或 ``layout`` / ``data_type`` 不受支持。
        AnnotationError: 目录缺失、标注缺失/非法、未知标签、扩展名冲突等扫描期数据问题。
    """
    spec.validate()
    if spec.layout == "folder_per_class":
        scan_subdir = _resolve_scan_subdir(spec, subdir)
        refs = _scan_folder_layout(spec, scan_subdir)
    elif spec.layout == "labelme":
        if subdir:
            raise DatasetSpecError(
                "subdir 只对 layout: folder_per_class 生效",
                hint="labelme 布局直接递归扫描 spec.root；请去掉 subdir 参数",
            )
        refs = _scan_labelme_layout(spec)
    else:  # pragma: no cover - spec.validate() 已限定枚举
        raise DatasetSpecError(
            f"不支持的 layout: {spec.layout!r}",
            hint="可选值: folder_per_class / labelme",
        )

    refs.sort(key=lambda ref: ref.rel_path)
    _ensure_unique_rel_paths(refs)
    return refs


# ---------------------------------------------------------------------------
# 布局分派（延迟 import：parsers 需要本模块的 SampleRef，避免循环导入）
# ---------------------------------------------------------------------------


def _scan_folder_layout(spec: DVXDatasetSpec, subdir: str | None) -> list[SampleRef]:
    """按 folder_per_class 布局扫描（多任务时逐任务扫描）。"""
    from dvisionix.data.parsers.folder import scan_folder_per_class

    refs: list[SampleRef] = []
    if spec.data_type == "multi_task_classification":
        for task in spec.tasks or []:
            refs.extend(scan_folder_per_class(spec.root, spec, task=task, subdir=subdir))
        return refs
    refs.extend(scan_folder_per_class(spec.root, spec, task=None, subdir=subdir))
    return refs


def _scan_labelme_layout(spec: DVXDatasetSpec) -> list[SampleRef]:
    """按 labelme 布局扫描（detection / segmentation）。"""
    from dvisionix.data.parsers.labelme import scan_labelme

    class_table = ClassTable(spec.class_names())
    return scan_labelme(spec.root, spec, class_table)


# ---------------------------------------------------------------------------
# dir 模式的公共子目录推断
# ---------------------------------------------------------------------------


def _resolve_scan_subdir(spec: DVXDatasetSpec, subdir: str | None) -> str | None:
    """返回本次 folder 扫描使用的额外子目录层（显式值优先，否则按 dir_map 推断）。"""
    if subdir is not None:
        return subdir
    splits = spec.splits
    if splits is None or splits.mode != "dir" or not splits.dir_map:
        return None
    derived = _common_subdir([value for value in splits.dir_map.values() if isinstance(value, str)])
    if not derived:
        return None
    bases = [os.path.join(spec.root, task.path) for task in (spec.tasks or [])]
    if not bases:
        bases = [spec.root]
    if not all(os.path.isdir(os.path.join(base, derived)) for base in bases):
        # 推断出的目录并非每个数据集目录下都存在：放弃推断，让 dir 切分去报告未覆盖样本
        return None
    print(
        f"[DVX] splits.mode=dir：仅扫描公共子目录 {derived!r}"
        "（多任务下相对每个 task.path），避免扁平类目录与切片目录重复计入",
        file=sys.stderr,
    )
    return derived


def _common_subdir(dir_values: list[str]) -> str | None:
    """求若干相对目录的公共一级子目录前缀（按路径分量比较，无公共前缀返回 None）。"""
    components = [_relative_components(value) for value in dir_values]
    if not components or any(not parts for parts in components):
        return None
    common: list[str] = []
    for position in range(min(len(parts) for parts in components)):
        candidates = {parts[position] for parts in components}
        if len(candidates) != 1:
            break
        common.append(components[0][position])
    return posixpath.join(*common) if common else None


def _relative_components(value: str) -> list[str]:
    """把相对目录拆成路径分量（posix 风格，去掉 ``./`` 与空段）。"""
    text = value.replace("\\", "/").strip()
    if not text or posixpath.isabs(text):
        return []
    return [part for part in text.split("/") if part not in ("", ".", "..")]


# ---------------------------------------------------------------------------
# 通用校验
# ---------------------------------------------------------------------------


def _ensure_unique_rel_paths(refs: list[SampleRef]) -> None:
    """要求 ``rel_path`` 全局唯一（切分/清单匹配依赖它定位样本）。"""
    seen: dict[str, SampleRef] = {}
    for ref in refs:
        previous = seen.get(ref.rel_path)
        if previous is not None:
            raise AnnotationError(
                f"扫描得到重复的 rel_path: {ref.rel_path}",
                hint=f"冲突样本: {previous.image_path} 与 {ref.image_path}；"
                "多任务数据集请确认各 task.path 互不重叠，且同一图片只出现一次",
            )
        seen[ref.rel_path] = ref
