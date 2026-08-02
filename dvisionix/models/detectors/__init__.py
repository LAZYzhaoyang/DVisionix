# -*- coding: utf-8 -*-
"""检测器子包。

- base：SingleStageDetector（backbone + neck + head 装配脚手架）。
- anchors：AnchorGenerator + bbox delta 编解码。
- fcos：FCOSDetector（anchor-free）。
- retinanet：RetinaNetDetector（anchor-based）。
- yolo：YOLODetector（YOLOv8 风格，anchor-free）。
"""

from .base import SingleStageDetector
from .anchors import AnchorGenerator, bbox2delta, delta2bbox
from .fcos import FCOSDetector
from .retinanet import RetinaNetDetector
from .yolo import YOLODetector
from .detr import DETRDetector

__all__ = [
    "SingleStageDetector",
    "AnchorGenerator",
    "bbox2delta",
    "delta2bbox",
    "FCOSDetector",
    "RetinaNetDetector",
    "YOLODetector",
    "DETRDetector",
]