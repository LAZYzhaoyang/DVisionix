# -*- coding: utf-8 -*-
"""检测损失子包。"""

from .assigner import GridAssigner
from .box_loss import GIoULoss, CIoULoss, L1BoxLoss
from .losses import ObjectnessLoss, GridDetectionLoss

__all__ = ["GridAssigner", "GIoULoss", "CIoULoss", "L1BoxLoss", "ObjectnessLoss", "GridDetectionLoss"]