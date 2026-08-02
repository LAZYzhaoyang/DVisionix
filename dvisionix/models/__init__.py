# -*- coding: utf-8 -*-
"""
模型模块

组件化架构：backbones / necks / heads / detectors / losses / postprocess。

- ``models.toy``：教学级模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel），
  仅用于演示与快速验证，生产请用组件化模型。
- ``models.detectors``：SingleStageDetector 脚手架 + FCOSDetector（anchor-free）+
  RetinaNetDetector（anchor-based）+ AnchorGenerator。
- ``models.losses``：Loss 组件（BaseLoss 继承 + LossComposer 组合 + 检测 assigner/损失）。
"""

from .base import BaseModel, TASK_TYPES
from .backbones import TimmBackbone, TimmClassifier, list_timm_models, SequentialBackbone
from .layers import (
    build_layer,
    build_norm_layer,
    build_activation_layer,
    ConvNormAct,
    MLP,
    SEBlock,
    DropPath,
    create_timm_layer,
    list_timm_layers,
)
from .postprocess import nms, batched_nms, box_iou, fcos_decode, retinanet_decode, yolo_decode, detr_decode
from .classifiers import LinearClassifier
from .segmenters import SegmentationModel
from .detectors import SingleStageDetector, AnchorGenerator, FCOSDetector, RetinaNetDetector, YOLODetector, DETRDetector
from .necks import FPN, PANet
from .heads import ClsHead, SegHead, FCNHead, DeepLabV3Head, UNetDecoder, SegFormerHead, MaskFormerHead, DetHead, ArcFaceHead, MultiLabelHead, CosFaceHead, SphereFaceHead, AdaFaceHead, FCOSHead, RetinaNetHead, YOLOHead, DETRHead
from . import toy
from .toy import SimpleCNN, SimpleSegmentationModel, GridDetectionModel
from . import losses
from .losses import (
    BaseLoss,
    LossComposer,
    build_loss,
    build_losses,
    compute_loss,
    CrossEntropy,
    FocalLoss,
    BinaryCrossEntropy,
    DiceLoss,
    CombinedSegmentationLoss,
    GridAssigner,
    GridDetectionLoss,
    ObjectnessLoss,
    GIoULoss,
    CIoULoss,
    L1BoxLoss,
    FCOSAssigner,
    MaxIoUAssigner,
    ATSSAssigner,
    TaskAlignedAssigner,
    SigmoidFocalLoss,
    FCOSDetectionLoss,
    RetinaNetLoss,
    DETRLoss,
    YOLOLoss,
)

__all__ = [
    "BaseModel",
    "TASK_TYPES",
    "TimmBackbone",
    "TimmClassifier",
    "SequentialBackbone",
    "list_timm_models",
    "build_layer",
    "build_norm_layer",
    "build_activation_layer",
    "ConvNormAct",
    "MLP",
    "SEBlock",
    "DropPath",
    "create_timm_layer",
    "list_timm_layers",
    "nms",
    "batched_nms",
    "box_iou",
    "fcos_decode",
    "retinanet_decode",
    "yolo_decode",
    "detr_decode",
    "LinearClassifier",
    "SegmentationModel",
    "SingleStageDetector",
    "AnchorGenerator",
    "FCOSDetector",
    "RetinaNetDetector",
    "YOLODetector",
    "DETRDetector",
    "FPN",
    "PANet",
    "ClsHead",
    "SegHead",
    "FCNHead",
    "DeepLabV3Head",
    "UNetDecoder",
    "SegFormerHead",
    "MaskFormerHead",
    "DetHead",
    "ArcFaceHead",
    "MultiLabelHead",
    "CosFaceHead",
    "SphereFaceHead",
    "AdaFaceHead",
    "FCOSHead",
    "RetinaNetHead",
    "YOLOHead",
    "DETRHead",
    "toy",
    # 教学级模型（re-export，兼容旧导入）
    "SimpleCNN",
    "SimpleSegmentationModel",
    "GridDetectionModel",
    # losses
    "BaseLoss",
    "LossComposer",
    "build_loss",
    "build_losses",
    "compute_loss",
    "CrossEntropy",
    "FocalLoss",
    "BinaryCrossEntropy",
    "DiceLoss",
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
]

# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict
from ..registry import MODELS, NECKS, HEADS, LAYERS


def build_neck(cfg: Dict[str, Any]):
    """从配置构建颈部。"""
    return NECKS.build(dict(cfg))


def build_head(cfg: Dict[str, Any]):
    """从配置构建头部。"""
    return HEADS.build(dict(cfg))


def build_model(cfg: Dict[str, Any]):
    """从配置构建模型。"""
    return MODELS.build(dict(cfg))


__all__ = __all__ + ["MODELS", "NECKS", "HEADS", "LAYERS", "build_model", "build_neck", "build_head"]