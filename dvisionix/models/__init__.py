# -*- coding: utf-8 -*-
"""
模型模块

组件化架构：backbones / necks / heads / detectors / losses / postprocess。

- ``models.toy``：教学级模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel），
  仅用于演示与快速验证，生产请用组件化模型。
- ``models.detectors``：SingleStageDetector 脚手架（backbone + neck + head 装配）+
  FCOS / RetinaNet / YOLO / DETR / RT-DETR 检测器 + AnchorGenerator；各检测器自带
  专属 decode（与 head 同文件，便于定位与维护）。
- ``models.losses``：Loss 组件（BaseLoss 继承 + LossComposer 组合 + 检测 assigner/损失）。
"""

from . import (
    losses,  # noqa: F401
    toy,
)
from .backbones import SequentialBackbone, TimmBackbone, TimmClassifier, list_timm_models
from .base import TASK_TYPES, BaseModel
from .classifiers import LinearClassifier
from .detectors import (
    AnchorGenerator,
    DeformableDETRDetector,
    DETRDetector,
    FCOSDetector,
    RetinaNetDetector,
    RTDETRDetector,
    SingleStageDetector,
    YOLODetector,
    detr_decode,
    fcos_decode,
    retinanet_decode,
    yolo_decode,
)
from .heads import (
    AdaFaceHead,
    ArcFaceHead,
    ClsHead,
    CosFaceHead,
    DeepLabV3Head,
    DetHead,
    DETRHead,
    FCNHead,
    FCOSHead,
    MaskFormerHead,
    MultiLabelHead,
    RetinaNetHead,
    SegFormerHead,
    SegHead,
    SphereFaceHead,
    UNetDecoder,
    YOLOHead,
    maskformer_decode,
)
from .layers import (
    MLP,
    ConvNormAct,
    DropPath,
    SEBlock,
    build_activation_layer,
    build_layer,
    build_norm_layer,
    create_timm_layer,
    list_timm_layers,
)
from .losses import (
    ATSSAssigner,
    BaseLoss,
    BinaryCrossEntropy,
    CIoULoss,
    CombinedSegmentationLoss,
    CrossEntropy,
    DETRLoss,
    DiceLoss,
    DistillationLoss,
    FCOSAssigner,
    FCOSDetectionLoss,
    FocalLoss,
    GIoULoss,
    GridAssigner,
    GridDetectionLoss,
    L1BoxLoss,
    LossComposer,
    MaskFormerLoss,
    MaxIoUAssigner,
    ObjectnessLoss,
    RetinaNetLoss,
    SigmoidFocalLoss,
    TaskAlignedAssigner,
    YOLOLoss,
    build_loss,
    build_losses,
    compute_loss,
)
from .necks import FPN, PANet
from .postprocess import batched_nms, box_iou, nms
from .segmenters import SegmentationModel
from .toy import GridDetectionModel, SimpleCNN, SimpleSegmentationModel

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
    "maskformer_decode",
    "LinearClassifier",
    "SegmentationModel",
    "SingleStageDetector",
    "AnchorGenerator",
    "FCOSDetector",
    "RetinaNetDetector",
    "YOLODetector",
    "DETRDetector",
    "RTDETRDetector",
    "DeformableDETRDetector",
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
    "RTDETRHead",
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
    "DistillationLoss",
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
]

# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict

from ..registry import HEADS, LAYERS, MODELS, NECKS


def build_neck(cfg: Dict[str, Any]):
    """从配置构建颈部。"""
    return NECKS.build(dict(cfg))


def build_head(cfg: Dict[str, Any]):
    """从配置构建头部。"""
    return HEADS.build(dict(cfg))


def build_model(cfg: Dict[str, Any]):
    """从配置构建模型。"""
    return MODELS.build(dict(cfg))


__all__ = __all__ + [
    "MODELS",
    "NECKS",
    "HEADS",
    "LAYERS",
    "build_model",
    "build_neck",
    "build_head",
]
