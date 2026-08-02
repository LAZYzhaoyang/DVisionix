# -*- coding: utf-8 -*-
"""检测损失子包。"""

from .assigner import (
    GridAssigner,
    FCOSAssigner,
    MaxIoUAssigner,
    ATSSAssigner,
    TaskAlignedAssigner,
)
from .box_loss import GIoULoss, CIoULoss, L1BoxLoss
from .losses import (
    DETRLoss,
    ObjectnessLoss,
    GridDetectionLoss,
    SigmoidFocalLoss,
    FCOSDetectionLoss,
    RetinaNetLoss,
    YOLOLoss,
)

__all__ = [
    "GridAssigner", "FCOSAssigner", "MaxIoUAssigner", "ATSSAssigner", "TaskAlignedAssigner",
    "GIoULoss", "CIoULoss", "L1BoxLoss",
    "ObjectnessLoss", "GridDetectionLoss", "SigmoidFocalLoss",
    "FCOSDetectionLoss", "RetinaNetLoss", "YOLOLoss", "DETRLoss",
]