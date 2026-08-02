# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 骨干网络模块
"""
骨干网络模块

提供基于 timm 的骨干网络封装（ResNet / ViT / Swin 等），
以及开箱即用的 timm 分类模型。
"""

from .convnext import ConvNeXtBackbone
from .convnextv2 import ConvNeXtV2Backbone
from .cspdarknet import CSPDarknetBackbone
from .efficientnet_lite import EfficientNetLiteBackbone
from .feature import FeatureBackboneBase
from .mit import MiTBackbone
from .mobilenetv3 import MobileNetV3Backbone
from .sequential import SequentialBackbone
from .swin import SwinBackbone
from .swinv2 import SwinV2Backbone
from .timm_backbone import TimmBackbone, TimmClassifier, list_timm_models
from .vit import ViTBackbone

__all__ = [
    "TimmBackbone",
    "TimmClassifier",
    "list_timm_models",
    "SequentialBackbone",
    "FeatureBackboneBase",
    "ConvNeXtBackbone",
    "CSPDarknetBackbone",
    "MobileNetV3Backbone",
    "ViTBackbone",
    "SwinBackbone",
    "ConvNeXtV2Backbone",
    "EfficientNetLiteBackbone",
    "MiTBackbone",
    "SwinV2Backbone",
]
