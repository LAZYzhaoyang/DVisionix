# -*- coding: utf-8 -*-
"""检测损失子包。"""

from .assigner import GridAssigner, FCOSAssigner, MaxIoUAssigner, ATSSAssigner
from .box_loss import GIoULoss, CIoULoss, L1BoxLoss
from .losses import (
    ObjectnessLoss,
    GridDetectionLoss,
    SigmoidFocalLoss,
    FCOSDetectionLoss,
    RetinaNetLoss,
)

__all__ = [
    "GridAssigner",
    "FCOSAssigner",
    "MaxIoUAssigner",
    "ATSSAssigner",
    "GIoULoss",
    "CIoULoss",
    "L1BoxLoss",
    "ObjectnessLoss",
    "GridDetectionLoss",
    "SigmoidFocalLoss",
    "FCOSDetectionLoss",
    "RetinaNetLoss",
]