# -*- coding: utf-8 -*-
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
    FCOSDetectionLoss,
    GridDetectionLoss,
    ObjectnessLoss,
    OneToOneYOLOLoss,
    RetinaNetLoss,
    SigmoidFocalLoss,
    YOLOLoss,
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
]
