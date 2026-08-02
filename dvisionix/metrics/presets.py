# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 各任务的预设指标组合 + 预设类范例。
"""各任务的预设指标组合 + 预设类范例。

两种入口：
- get_preset_metrics(task_type, num_classes=...)：函数式快捷入口，返回 MetricCollection。
- ClassificationMetrics / SegmentationMetrics / DetectionMetrics：预设类，
  它们既是"开箱即用的默认组合"，也是"如何基于 BaseMetric + 原子指标封装自定义
  Metrics 类"的参考范例——使用者可仿照它们实现自己的组合指标。
"""

from typing import Any, Dict, List, Optional

from ..registry import METRICS
from .base import BaseMetric
from .classification import Accuracy, F1Score, Precision, Recall
from .collection import MetricCollection
from .detection import MeanAveragePrecision
from .segmentation import MeanIoU, PixelAccuracy


def _classification_metrics(
    num_classes: Optional[int] = None, average: str = "macro"
) -> List[BaseMetric]:
    return [
        Accuracy(),
        Precision(num_classes=num_classes, average=average),
        Recall(num_classes=num_classes, average=average),
        F1Score(num_classes=num_classes, average=average),
    ]


def _segmentation_metrics(num_classes: int, ignore_index: Optional[int] = 255) -> List[BaseMetric]:
    return [
        MeanIoU(num_classes=num_classes, ignore_index=ignore_index),
        PixelAccuracy(num_classes=num_classes, ignore_index=ignore_index),
    ]


def _detection_metrics(num_classes: int, **kwargs: Any) -> List[BaseMetric]:
    return [MeanAveragePrecision(num_classes=num_classes, **kwargs)]


def get_preset_metrics(
    task_type: str, num_classes: Optional[int] = None, **kwargs: Any
) -> MetricCollection:
    """返回某任务的默认指标组合（MetricCollection）。

    Args:
        task_type: 'classification' / 'segmentation' / 'detection'。
        num_classes: 类别数（分割/检测必填，分类可选）。
        **kwargs: 传给具体指标的额外参数（如 average / ignore_index / use_torchmetrics）。
    """
    if task_type == "classification":
        average = kwargs.pop("average", "macro")
        return MetricCollection(_classification_metrics(num_classes, average))
    if task_type == "segmentation":
        ignore_index = kwargs.pop("ignore_index", 255)
        return MetricCollection(_segmentation_metrics(num_classes, ignore_index))
    if task_type == "detection":
        return MetricCollection(_detection_metrics(num_classes, **kwargs))
    raise ValueError(f"Unknown task type: {task_type!r}")


class _PresetMetric(BaseMetric):
    """预设组合类的基类：内部持有一个 MetricCollection，并把接口委托给它。

    这是"基于 BaseMetric 封装组合指标"的参考实现，子类只需在 _build_metrics
    中返回原子指标列表即可。
    """

    def __init__(self, name: str, metrics: List[BaseMetric]):
        self._collection = MetricCollection(metrics)
        super().__init__(name)

    def reset(self) -> None:
        # BaseMetric.__init__ 会在设置 _collection 后调用本方法
        """重置预设指标集合。"""
        self._collection.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """用 (preds, targets) 更新预设指标集合。"""
        self._collection.update(*args, **kwargs)

    def compute(self) -> Dict[str, Any]:
        """计算并返回预设指标集合的结果字典。"""
        return self._collection.compute()


@METRICS.register()
@METRICS.register(name="classification_metrics")
class ClassificationMetrics(_PresetMetric):
    """分类默认指标组合：accuracy / precision / recall / f1。

    也是自定义组合指标的范例：继承 _PresetMetric，传入原子指标列表即可。
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: str = "macro",
        name: str = "classification",
    ):
        super().__init__(name, _classification_metrics(num_classes, average))


@METRICS.register()
@METRICS.register(name="segmentation_metrics")
class SegmentationMetrics(_PresetMetric):
    """分割默认指标组合：mIoU / pixel_accuracy。"""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = 255,
        name: str = "segmentation",
        **kwargs: Any,
    ):
        super().__init__(name, _segmentation_metrics(num_classes, ignore_index))


@METRICS.register()
@METRICS.register(name="detection_metrics")
class DetectionMetrics(_PresetMetric):
    """检测默认指标组合：mAP / mAP_50 / mAP_75。"""

    def __init__(self, num_classes: int, name: str = "detection", **kwargs: Any):
        super().__init__(name, _detection_metrics(num_classes, **kwargs))


__all__ = [
    "get_preset_metrics",
    "ClassificationMetrics",
    "SegmentationMetrics",
    "DetectionMetrics",
]
