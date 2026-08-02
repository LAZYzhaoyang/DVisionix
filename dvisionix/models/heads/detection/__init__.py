# -*- coding: utf-8 -*-
"""检测头子包。"""

from .grid import DetHead
from .fcos import FCOSHead
from .retinanet import RetinaNetHead
from .yolo import YOLOHead
from .detr import DETRHead

__all__ = ["DetHead", "FCOSHead", "RetinaNetHead", "YOLOHead", "DETRHead"]