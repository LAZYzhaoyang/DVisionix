# -*- coding: utf-8 -*-
"""检测头子包。"""

from .detr import DETRHead
from .fcos import FCOSHead
from .grid import DetHead
from .retinanet import RetinaNetHead
from .rtdetr import RTDETRHead
from .yolo import YOLOHead

__all__ = ["DetHead", "FCOSHead", "RetinaNetHead", "YOLOHead", "DETRHead", "RTDETRHead"]
