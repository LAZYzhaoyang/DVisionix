# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 指标模块。
"""指标模块。

设计：原子指标（可自由组合）+ MetricCollection（组合容器）+ 预设组合。

- 原子指标：Accuracy / TopKAccuracy / Precision / Recall / F1Score（分类），
  MeanIoU / PixelAccuracy / DiceScore（分割），MeanAveragePrecision（检测）。
- 组合：MetricCollection([...]) 接受实例 / 配置字典 / 字符串，可混用。
- 预设：get_preset_metrics(task_type, num_classes=...) 或
  ClassificationMetrics / SegmentationMetrics / DetectionMetrics（预设组合 + 自定义范例）。
- 所有指标都注册到全局 METRICS 注册表，可配置驱动构建。
"""

from .base import BaseMetric
from .classification import Accuracy, F1Score, Precision, Recall, TopKAccuracy
from .collection import MetricCollection, build_single_metric
from .detection import MeanAveragePrecision
from .panoptic import PanopticQuality
from .presets import (
    ClassificationMetrics,
    DetectionMetrics,
    SegmentationMetrics,
    get_preset_metrics,
)
from .segmentation import DiceScore, MaskAveragePrecision, MeanIoU, PixelAccuracy

__all__ = [
    "BaseMetric",
    "Accuracy",
    "TopKAccuracy",
    "Precision",
    "Recall",
    "F1Score",
    "MeanIoU",
    "PixelAccuracy",
    "MaskAveragePrecision",
    "PanopticQuality",
    "DiceScore",
    "MeanAveragePrecision",
    "MetricCollection",
    "build_single_metric",
    "get_preset_metrics",
    "ClassificationMetrics",
    "SegmentationMetrics",
    "DetectionMetrics",
]


# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict

from ..registry import METRICS


def build_metric(cfg: Dict[str, Any]):
    """从配置构建单个指标（等价于 METRICS.build）。"""
    return METRICS.build(dict(cfg))


__all__ = __all__ + ["METRICS", "build_metric"]
