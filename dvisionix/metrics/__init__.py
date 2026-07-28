# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\metrics\\__init__.py

"""
指标模块

提供各种任务的评估指标计算。
"""

from .classification import ClassificationMetrics
from .segmentation import SegmentationMetrics
from .detection import DetectionMetrics
from .collection import MetricCollection

__all__ = [
    "ClassificationMetrics",
    "SegmentationMetrics",
    "DetectionMetrics",
    "MetricCollection",
]


# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict
from ..registry import METRICS

for _cls in (ClassificationMetrics, SegmentationMetrics, DetectionMetrics, MetricCollection):
    if _cls.__name__ not in METRICS:
        METRICS.register(_cls)


def build_metric(cfg):
    """从配置构建指标计算器。"""
    return METRICS.build(dict(cfg))


__all__ = __all__ + ["METRICS", "build_metric"]
