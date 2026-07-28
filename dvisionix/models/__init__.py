# -*- coding: utf-8 -*-
"""
模型模块

提供模型基类和各种任务的示例模型，以及 backbone/neck/head 组件化架构。
"""

from .base import (
    BaseModel,
    SimpleCNN,
    SimpleSegmentationModel,
    TASK_TYPES,
)
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
from .detection import GridDetectionModel
from .postprocess import nms, batched_nms, box_iou
from .necks import FPN
from .heads import ClsHead, SegHead, DetHead
from .detectors import GeneralizedModel

__all__ = [
    "BaseModel",
    "SimpleCNN",
    "SimpleSegmentationModel",
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
    "GridDetectionModel",
    "nms",
    "batched_nms",
    "box_iou",
    "FPN",
    "ClsHead",
    "SegHead",
    "DetHead",
    "GeneralizedModel",
]

# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict
from ..registry import MODELS, NECKS, HEADS, LAYERS

_MODEL_ALIASES = {
    SimpleCNN: "simple_cnn",
    SimpleSegmentationModel: "simple_segmentation",
    GridDetectionModel: "grid_detection",
    TimmClassifier: "timm_classifier",
}
# 新增组件注册
for _cls in (FPN,):
    if _cls.__name__ not in NECKS:
        NECKS.register(_cls)
for _cls in (ClsHead, SegHead, DetHead):
    if _cls.__name__ not in HEADS:
        HEADS.register(_cls)
if GeneralizedModel.__name__ not in MODELS:
    MODELS.register(GeneralizedModel)
if "generalized" not in MODELS:
    MODELS.register(GeneralizedModel, name="generalized")
for _cls in (SimpleCNN, SimpleSegmentationModel, GridDetectionModel, TimmClassifier):
    if _cls.__name__ not in MODELS:
        MODELS.register(_cls)
for _cls, _alias in _MODEL_ALIASES.items():
    if _alias not in MODELS:
        MODELS.register(_cls, name=_alias)


def build_neck(cfg: Dict[str, Any]):
    """从配置构建颈部。"""
    return NECKS.build(dict(cfg))


def build_head(cfg: Dict[str, Any]):
    """从配置构建头部。"""
    return HEADS.build(dict(cfg))


def build_model(cfg: Dict[str, Any]):
    """从配置构建模型。

    例如::

        build_model({"type": "SimpleCNN", "num_classes": 10})
        build_model({"type": "TimmClassifier", "name": "resnet18", "num_classes": 10})
    """
    return MODELS.build(dict(cfg))


__all__ = __all__ + ["MODELS", "NECKS", "HEADS", "LAYERS", "build_model", "build_neck", "build_head"]
