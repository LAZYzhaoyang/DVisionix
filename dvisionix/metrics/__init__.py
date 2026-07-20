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
