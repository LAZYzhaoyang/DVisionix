# -*- coding: utf-8 -*-
"""检测器子包。

- base：SingleStageDetector（backbone + neck + head 装配脚手架）。
- anchors：AnchorGenerator + bbox delta 编解码。
- fcos：FCOSDetector（anchor-free）。
- retinanet：RetinaNetDetector（anchor-based）。
- yolo：YOLODetector（YOLOv8 风格，anchor-free）。
"""

from ..layers.anchors import AnchorGenerator, bbox2delta, delta2bbox
from .base import SingleStageDetector, detr_decode
from .centernet import CenterNetDetector
from .deformable_detr import DeformableDETRDetector
from .detr import DETRDetector
from .dino import DINODetrDetector
from .fcos import FCOSDetector, fcos_decode
from .nmsfree_yolo import NMSFreeYOLODetector
from .retinanet import RetinaNetDetector, retinanet_decode
from .rtdetr import RTDETRDetector
from .rtdetr_full import RTDETRFullDetector
from .yolo import YOLODetector, yolo_decode
from .yolov9 import YOLOv9Detector

__all__ = [
    "SingleStageDetector",
    "AnchorGenerator",
    "bbox2delta",
    "delta2bbox",
    "FCOSDetector",
    "RetinaNetDetector",
    "YOLODetector",
    "detr_decode",
    "fcos_decode",
    "retinanet_decode",
    "yolo_decode",
    "DETRDetector",
    "RTDETRDetector",
    "RTDETRFullDetector",
    "DeformableDETRDetector",
    "CenterNetDetector",
    "NMSFreeYOLODetector",
    "DINODetrDetector",
    "YOLOv9Detector",
]
