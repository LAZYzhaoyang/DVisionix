# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 标准数据集（dvx_dataset.yaml -> 按 split 输出的 Dataset，契约 P1-7）。
"""DVX 标准数据集（``DVXDataset``，契约 P1-7）。

把一份 ``dvx_dataset.yaml`` 规格变成可直接喂给 ``DataLoader`` 的 ``Dataset``：

1. 构造期：``scan_samples(spec)`` 扫描全部样本 -> ``build_splits`` 得到切分索引
   （调用方也可直接传入 ``split_result`` 复用既有切分）-> 取当前 ``split`` 的下标；
2. 取样本：按 ``SampleRef`` 组出符合 ``Sample`` 契约的 dict，再**复用**
   ``BaseDataset`` 的流程：``self.load_image``（cv2 读图）-> ``self._check_sample_keys``
   -> ``self.transforms``。

四种 ``data_type`` 的输出字段：

========================  ======================================
data_type                 样本字段
========================  ======================================
classification            ``image``、``label``
multi_task_classification ``image``、``label``、``task``
detection                 ``image``、``boxes``、``labels``
segmentation              ``image``、``mask``
========================  ======================================

规范（契约 P1-7 / 附二）：

- 检测：``boxes`` 为 ``(N, 4)`` ``float32`` xyxy 绝对像素，``labels`` 为 ``(N,)`` ``int64``；
  空标注是合法的负样本（``(0, 4)`` / ``(0,)``）。
- 分割：``mask`` 为 ``(H, W)`` ``int64``，由 ``SampleRef.segments`` /
  ``segment_labels`` 用 ``polygons_to_mask(..., background=0)`` 栅格化，尺寸取
  ``SampleRef.image_size``（labelme 的 ``imageHeight`` / ``imageWidth``，缺失时才读图兜底）。
  真实分割数据集单张 1K 掩码约 8 MiB，因此掩码在**取样本时按需生成、不常驻**，
  避免构造期把整个 split 的掩码同时放进内存。
- 多任务分类：``classes`` / ``num_classes`` 没有统一值（各任务有独立类别表），
  因此为 ``None``；任务列表见 ``task_names``，某任务的类别数请用 ``spec.num_classes(task)``。

用法::

    from dvisionix.data import DVXDataset, load_spec

    spec = load_spec("project/data/det/dvx_dataset.yaml")
    train = DVXDataset(spec, split="train", transforms=my_pipeline)
    sample = train[0]  # {"image": ndarray, "boxes": (N,4) float32, "labels": (N,) int64}
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ..errors import DatasetError, DatasetSpecError, SplitError
from .base import BaseDataset
from .collate import detection_collate, segmentation_collate
from .parsers.rasterize import polygons_to_mask
from .spec import DVXDatasetSpec, SampleRef, SplitResult, build_splits, load_spec, scan_samples

__all__ = ["DVXDataset"]


class DVXDataset(BaseDataset):
    """按 ``dvx_dataset.yaml`` 构建、按 ``split`` 切片的数据集（契约 P1-7）。

    Args:
        spec: 规格对象，或 ``dvx_dataset.yaml`` 的路径（``str`` / ``os.PathLike``）。
        split: 要取的切分名（通常在 ``{"train", "val", "test"}`` 中）。
        transforms: ``TransformPipeline`` 或任意 ``sample -> sample`` 可调用对象；
            为 None 时只把 ``image`` 读成 numpy。
        task: 仅 ``multi_task_classification`` 可用；给定后只保留该任务的样本。
        split_result: 已有的切分结果；给定时直接使用，不再调用 ``build_splits``
            （下标指向本 dataset 扫描得到的全量 ``refs``）。

    Raises:
        DatasetSpecError: ``spec`` 类型非法、spec 校验失败、``task`` 用错数据集
            或任务名不存在。
        AnnotationError: 扫描期数据问题（缺标注、未知类别、扩展名冲突等）。
        SplitError: 缺少切分配置、``split`` 不在切分结果中、``split_result`` 下标越界。
    """

    def __init__(
        self,
        spec: "str | os.PathLike[str] | DVXDatasetSpec",
        split: str = "train",
        transforms: Any | None = None,
        task: str | None = None,
        split_result: SplitResult | None = None,
    ) -> None:
        resolved = _resolve_spec(spec)
        resolved.validate()
        self._check_split_name(split)
        self._check_task_argument(resolved, task)

        all_refs = scan_samples(resolved)
        result = (
            split_result
            if split_result is not None
            else build_splits(all_refs, resolved.splits, resolved)
        )
        if not isinstance(result, SplitResult):
            raise SplitError(
                f"split_result 必须是 SplitResult（当前为 {type(result).__name__}）",
                hint="用 build_splits(...) 或 SplitResult.load(path) 得到切分结果",
            )
        indices = _split_indices(result, split, len(all_refs))

        refs = [all_refs[index] for index in indices]
        if task is not None:
            refs = [ref for ref in refs if ref.task == task]
            if not refs:
                warnings.warn(
                    f"split {split!r} 中任务 {task!r} 没有样本：数据集为空",
                    UserWarning,
                    stacklevel=2,
                )

        self._spec = resolved
        self._split = split
        self._task = task
        self._split_result = result
        self._refs = refs

        samples = [self._sample_stub(ref) for ref in refs]
        super().__init__(
            samples,
            transforms=transforms,
            collate_fn=_default_collate(resolved.data_type),
        )
        # 供上层按任务取预设 pipeline / collate（与 CustomDataset 的约定一致）
        self.task_type = resolved.data_type

    # ------------------------------------------------------------------
    # 只读属性（契约 P1-7）
    # ------------------------------------------------------------------

    @property
    def spec(self) -> DVXDatasetSpec:
        """构建本数据集所用的规格对象。"""
        return self._spec

    @property
    def split(self) -> str:
        """当前切分名。"""
        return self._split

    @property
    def data_type(self) -> str:
        """数据类型：classification / multi_task_classification / detection / segmentation。"""
        return self._spec.data_type

    @property
    def classes(self) -> list[str] | None:
        """单任务类别表；多任务分类没有统一类别表，返回 None。"""
        if self.data_type == "multi_task_classification":
            return None
        return self._spec.class_names()

    @property
    def task_names(self) -> list[str] | None:
        """多任务分类的任务名列表；其它数据类型返回 None。"""
        if self.data_type != "multi_task_classification":
            return None
        return [task.name for task in (self._spec.tasks or [])]

    @property
    def num_classes(self) -> int | None:
        """单任务类别数；多任务分类返回 None（请用 ``spec.num_classes(task)``）。"""
        names = self.classes
        return len(names) if names is not None else None

    @property
    def refs(self) -> list[SampleRef]:
        """当前 split（若指定了 ``task`` 则再按任务过滤）的样本引用列表。"""
        return list(self._refs)

    @property
    def split_result(self) -> SplitResult:
        """本数据集实际使用的切分结果。"""
        return self._split_result

    # ------------------------------------------------------------------
    # Dataset 协议
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        """返回一个符合 ``Sample`` 契约的样本 dict。

        流程与 ``BaseDataset.__getitem__`` 保持一致：组样本 -> ``load_image``
        -> ``_check_sample_keys`` -> ``transforms``。分割掩码在这里按 ``image_size``
        栅格化（不缓存，避免整个 split 的 (H, W) int64 掩码常驻内存）。
        """
        ref = self._refs[index]
        sample = dict(self.samples[index])
        if self.data_type == "segmentation":
            sample["mask"] = self._rasterize(ref)
        sample["image"] = self.load_image(sample)
        self._check_sample_keys(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    # ------------------------------------------------------------------
    # 样本组装
    # ------------------------------------------------------------------

    def _sample_stub(self, ref: SampleRef) -> dict:
        """构造不依赖 transforms 的样本骨架（image 为路径，分割暂不放 mask）。"""
        data_type = self.data_type
        sample: dict[str, Any] = {"image": ref.image_path}
        if data_type == "classification":
            sample["label"] = _require_label(ref)
        elif data_type == "multi_task_classification":
            sample["label"] = _require_label(ref)
            if not isinstance(ref.task, str) or not ref.task:
                raise DatasetError(
                    f"多任务样本缺少 task 字段: {ref.rel_path}",
                    hint="scan_samples 会为多任务分类写入 task；请检查 spec.tasks 与 task.path",
                )
            sample["task"] = ref.task
        elif data_type == "detection":
            sample["boxes"], sample["labels"] = _detection_arrays(ref)
        elif data_type == "segmentation":
            # mask 在 __getitem__ 里按 image_size 栅格化（构造期不预生成整份 split）
            pass
        else:  # pragma: no cover - spec.validate() 已限定枚举
            raise DatasetError(
                f"不支持的 data_type: {data_type!r}",
                hint="可选值: classification / multi_task_classification / detection / segmentation",
            )
        return sample

    def _rasterize(self, ref: SampleRef) -> np.ndarray:
        """把 ``SampleRef`` 的多边形栅格化为 ``(H, W)`` ``int64`` 掩码（背景 0）。"""
        segments = list(ref.segments or [])
        if ref.segment_labels is None:
            segment_labels: np.ndarray = np.zeros((0,), dtype=np.int64)
        else:
            segment_labels = np.asarray(ref.segment_labels, dtype=np.int64)
        height, width = self._mask_size(ref)
        return polygons_to_mask(segments, segment_labels, height, width, background=0)

    @staticmethod
    def _mask_size(ref: SampleRef) -> tuple[int, int]:
        """返回掩码尺寸 ``(H, W)``；``image_size`` 缺失时读图兜底并显式告警。"""
        if ref.image_size is not None:
            height, width = ref.image_size
            return int(height), int(width)
        image = cv2.imread(ref.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise DatasetError(
                f"无法确定分割掩码尺寸: {ref.image_path}",
                hint="labelme JSON 缺 imageWidth/imageHeight 且图片无法读取；"
                "请检查图片文件或补全 JSON 的尺寸字段",
            )
        size = (int(image.shape[0]), int(image.shape[1]))
        warnings.warn(
            f"样本缺少 image_size，已用图片实际尺寸 {size} 兜底: {ref.rel_path}",
            UserWarning,
            stacklevel=3,
        )
        return size

    # ------------------------------------------------------------------
    # 入参校验
    # ------------------------------------------------------------------

    @staticmethod
    def _check_split_name(split: Any) -> None:
        """要求 ``split`` 是非空字符串。"""
        if not isinstance(split, str) or not split.strip():
            raise SplitError(
                f"split 必须是非空字符串（当前为 {split!r}）",
                hint="例如 split='train' / 'val' / 'test'",
            )

    @staticmethod
    def _check_task_argument(spec: DVXDatasetSpec, task: Any) -> None:
        """校验 ``task`` 参数的适用范围与任务名。"""
        if task is None:
            return
        if spec.data_type != "multi_task_classification":
            raise DatasetSpecError(
                f"task 参数仅 multi_task_classification 可用（当前 data_type={spec.data_type!r}）",
                hint="单标签 / 检测 / 分割数据集请传 task=None",
            )
        names = [item.name for item in (spec.tasks or [])]
        if task not in names:
            raise DatasetSpecError(
                f"未知任务: {task!r}",
                hint=f"可选任务: {names}",
            )


# ---------------------------------------------------------------------------
# 模块级工具
# ---------------------------------------------------------------------------


def _resolve_spec(spec: "str | os.PathLike[str] | DVXDatasetSpec") -> DVXDatasetSpec:
    """把 ``str`` / ``os.PathLike`` / ``DVXDatasetSpec`` 统一成规格对象。"""
    if isinstance(spec, DVXDatasetSpec):
        return spec
    if isinstance(spec, (str, os.PathLike)):
        return load_spec(os.fspath(spec))
    raise DatasetSpecError(
        f"spec 必须是 dvx_dataset.yaml 路径或 DVXDatasetSpec（当前为 {type(spec).__name__}）",
        hint="例如 DVXDataset('project/data/cls/dvx_dataset.yaml') 或先 load_spec(...)",
    )


def _split_indices(result: SplitResult, split: str, total: int) -> list[int]:
    """取出某个 split 的下标并做边界/类型/重复校验，返回升序列表。"""
    available = list(result.splits)
    if split not in result.splits:
        raise SplitError(
            f"切分结果中没有 split {split!r}",
            hint=f"可用切分: {available}；请检查 dvx_dataset.yaml 的 splits 配置",
        )
    raw_indices = result.splits[split]
    if isinstance(raw_indices, (str, bytes, bytearray)) or not isinstance(raw_indices, Sequence):
        raise SplitError(
            f"split {split!r} 的下标必须是列表（当前为 {type(raw_indices).__name__}）",
            hint="期望 {split: [下标, ...]}；请用 SplitResult.save() 重新生成",
        )
    checked: list[int] = []
    for index in raw_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise SplitError(
                f"split {split!r} 含非整数下标: {index!r}",
                hint="下标必须是 Python 整数（由 build_splits 生成）",
            )
        if index < 0 or index >= total:
            raise SplitError(
                f"split {split!r} 含越界下标: {index}（扫描到 {total} 个样本）",
                hint="split_result 与当前 spec 扫描结果不匹配；请重新 build_splits 或确认数据未被改动",
            )
        checked.append(index)
    if len(checked) != len(set(checked)):
        raise SplitError(
            f"split {split!r} 含重复下标",
            hint="同一样本不能在一个 split 中出现两次；请重新生成切分结果",
        )
    return sorted(checked)


def _require_label(ref: SampleRef) -> int:
    """读取分类标签；缺失说明扫描结果与 data_type 不匹配。"""
    if ref.label is None:
        raise DatasetError(
            f"样本缺少分类标签: {ref.rel_path}",
            hint="分类样本应由扫描器写入 label；请检查 layout / classes 与目录结构是否匹配",
        )
    return int(ref.label)


def _detection_arrays(ref: SampleRef) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(N, 4) float32`` 框与 ``(N,) int64`` 类别（空标注给空数组）。"""
    if ref.boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = np.asarray(ref.boxes, dtype=np.float32).reshape(-1, 4).copy()
    if ref.labels is None:
        labels = np.zeros((0,), dtype=np.int64)
    else:
        labels = np.asarray(ref.labels, dtype=np.int64).reshape(-1).copy()
    if boxes.shape[0] != labels.shape[0]:
        raise DatasetError(
            f"检测样本的 boxes（{boxes.shape[0]} 个）与 labels（{labels.shape[0]} 个）数量不一致: "
            f"{ref.rel_path}",
            hint="请检查 labelme 解析结果；boxes 与 labels 必须一一对应",
        )
    return boxes, labels


def _default_collate(data_type: str) -> Any:
    """按数据类型返回默认 collate：检测/分割用变长/堆叠版本，分类用 None。"""
    if data_type == "detection":
        return detection_collate
    if data_type == "segmentation":
        return segmentation_collate
    return None
