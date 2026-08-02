# -*- coding: utf-8 -*-
"""Loss 组件（模型层的一部分）。

- ``BaseLoss``：自定义损失基类，继承并实现 ``forward(preds, targets, **kwargs)``。
- ``LossComposer``：多损失加权自由组合。
- ``build_loss / build_losses``：配置驱动构建（Registry: LOSSES）。
- 分任务损失：classification / segmentation / detection。
"""

from .base import BaseLoss, LossComposer, build_loss, build_losses, compute_loss
from .classification import (
    BinaryCrossEntropy,
    CircleLoss,
    CrossEntropy,
    DistillationLoss,
    FeatureDistillLoss,
    FocalLoss,
    InfoNCELoss,
)
from .detection import (
    ATSSAssigner,
    CenterNetLoss,
    CIoULoss,
    DETRLoss,
    DINOLoss,
    FCOSAssigner,
    FCOSDetectionLoss,
    GIoULoss,
    GridAssigner,
    GridDetectionLoss,
    L1BoxLoss,
    MaxIoUAssigner,
    ObjectnessLoss,
    OneToOneYOLOLoss,
    RetinaNetLoss,
    SigmoidFocalLoss,
    TaskAlignedAssigner,
    YOLOLoss,
    YOLOv9Loss,
)
from .segmentation import CombinedSegmentationLoss, DiceLoss, MaskFormerLoss

__all__ = [
    "BaseLoss",
    "LossComposer",
    "build_loss",
    "build_losses",
    "compute_loss",
    "CrossEntropy",
    "FocalLoss",
    "BinaryCrossEntropy",
    "DistillationLoss",
    "FeatureDistillLoss",
    "CircleLoss",
    "InfoNCELoss",
    "DiceLoss",
    "MaskFormerLoss",
    "CombinedSegmentationLoss",
    "GridAssigner",
    "GridDetectionLoss",
    "ObjectnessLoss",
    "GIoULoss",
    "CIoULoss",
    "L1BoxLoss",
    "FCOSAssigner",
    "MaxIoUAssigner",
    "ATSSAssigner",
    "TaskAlignedAssigner",
    "SigmoidFocalLoss",
    "FCOSDetectionLoss",
    "RetinaNetLoss",
    "DETRLoss",
    "YOLOLoss",
    "OneToOneYOLOLoss",
    "CenterNetLoss",
    "YOLOv9Loss",
    "DINOLoss",
]
