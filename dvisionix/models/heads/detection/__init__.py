# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测头子包。
"""检测头子包。"""

from .centernet import CenterNetHead
from .deformable_detr import DeformableDETRHead
from .detr import DETRHead
from .dino import DINODetrHead
from .fcos import FCOSHead
from .nmsfree_yolo import NMSFreeYOLOHead
from .retinanet import RetinaNetHead
from .rtdetr import RTDETRHead
from .rtdetr_full import RTDETRFullHead
from .yolo import YOLOHead

__all__ = [
    "FCOSHead",
    "RetinaNetHead",
    "YOLOHead",
    "DETRHead",
    "RTDETRHead",
    "RTDETRFullHead",
    "DeformableDETRHead",
    "CenterNetHead",
    "NMSFreeYOLOHead",
    "DINODetrHead",
]
