# -*- coding: utf-8 -*-
"""检测器子包。

- base：SingleStageDetector（backbone + neck + head 装配脚手架）。
- anchors：AnchorGenerator + bbox delta 编解码。
- fcos：FCOSDetector（anchor-free 单阶段）。
- retinanet：RetinaNetDetector（anchor-based 单阶段）。
"""

from .base import SingleStageDetector
from .anchors import AnchorGenerator, bbox2delta, delta2bbox
from .fcos import FCOSDetector
from .retinanet import RetinaNetDetector

__all__ = [
    "SingleStageDetector",
    "AnchorGenerator",
    "bbox2delta",
    "delta2bbox",
    "FCOSDetector",
    "RetinaNetDetector",
]