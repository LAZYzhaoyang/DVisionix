# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测损失子包。
"""检测损失子包。"""

from .assigner import (
    ATSSAssigner,
    FCOSAssigner,
    GridAssigner,
    MaxIoUAssigner,
    TaskAlignedAssigner,
)
from .box_loss import CIoULoss, GIoULoss, L1BoxLoss
from .losses import (
    CenterNetLoss,
    DETRLoss,
    DINOLoss,
    FCOSDetectionLoss,
    GridDetectionLoss,
    ObjectnessLoss,
    OneToOneYOLOLoss,
    RetinaNetLoss,
    SigmoidFocalLoss,
    YOLOLoss,
    YOLOv9Loss,
)

__all__ = [
    "GridAssigner",
    "FCOSAssigner",
    "MaxIoUAssigner",
    "ATSSAssigner",
    "TaskAlignedAssigner",
    "GIoULoss",
    "CIoULoss",
    "L1BoxLoss",
    "ObjectnessLoss",
    "GridDetectionLoss",
    "SigmoidFocalLoss",
    "FCOSDetectionLoss",
    "RetinaNetLoss",
    "YOLOLoss",
    "DETRLoss",
    "OneToOneYOLOLoss",
    "CenterNetLoss",
    "YOLOv9Loss",
    "DINOLoss",
]
