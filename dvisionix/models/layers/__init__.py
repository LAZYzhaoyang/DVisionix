# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 层（layer）模块。
"""层（layer）模块。

提供:
- 内置自定义层: ConvNormAct / MLP / SEBlock / DropPath（见 basic.py）。
- norm/激活的按名构建工具: build_norm_layer / build_activation_layer（见 builder.py）。
- timm 层封装: create_timm_layer / list_timm_layers（见 timm_layers.py）。
- 配置驱动构建: build_layer({"type": "conv_norm_act", ...})。

所有内置层都注册到全局 ``dvisionix.registry.LAYERS``，用户可仿照 basic.py 定义并注册自己的层。
"""

from typing import Any, Dict

from ...registry import LAYERS
from .anchors import AnchorGenerator, bbox2delta, delta2bbox
from .attention import PositionEmbeddingSine
from .basic import MLP, ConvNormAct, DropPath, SEBlock
from .builder import build_activation_layer, build_norm_layer
from .c3k2 import C3k2Block
from .convnext import ConvNeXtBlock
from .convnextv2 import ConvNeXtV2Block
from .csp import CSPLayer
from .deformable_attention import MultiScaleDeformableAttention, MultiScaleDeformableAttentionV2
from .detr_denoising import DenoisingQueryGenerator
from .elan import EELANLayer, ELANLayer
from .grn import GRN
from .mbconv import MBConvBlock
from .norm import LayerNorm2d
from .patch_ops import PatchExpand, PatchMerging
from .psa import PSABlock
from .query_selection import QuerySelection
from .relative_position_bias import ContinuousRelativePositionBias
from .reversible import ReversibleBlock
from .swinv2 import SwinV2Block
from .timm_layers import create_timm_layer, list_timm_layers
from .transformer import DeformableDecoderLayer, DeformableEncoderLayer, MixFFN
from .window_attention import WindowAttention, window_partition, window_reverse


def build_layer(cfg: Dict[str, Any]):
    """从配置字典构建层。

    Examples:
        >>> build_layer({"type": "conv_norm_act", "in_channels": 3, "out_channels": 16})
        >>> build_layer({"type": "se", "channels": 64, "reduction": 8})
        >>> build_layer({"type": "timm_squeeze_excite", "channels": 64})
    """
    return LAYERS.build(dict(cfg))


__all__ = [
    "LAYERS",
    "build_layer",
    "build_norm_layer",
    "build_activation_layer",
    "ConvNormAct",
    "MLP",
    "SEBlock",
    "DropPath",
    "CSPLayer",
    "ELANLayer",
    "EELANLayer",
    "MultiScaleDeformableAttention",
    "ConvNeXtBlock",
    "MBConvBlock",
    "ReversibleBlock",
    "create_timm_layer",
    "list_timm_layers",
]
__all__ = __all__ + [
    "PositionEmbeddingSine",
    "AnchorGenerator",
    "bbox2delta",
    "delta2bbox",
    "LayerNorm2d",
    "DeformableEncoderLayer",
    "DeformableDecoderLayer",
    "MixFFN",
    "WindowAttention",
    "window_partition",
    "window_reverse",
    "PatchMerging",
    "PatchExpand",
    "GRN",
    "QuerySelection",
    "DenoisingQueryGenerator",
    "ContinuousRelativePositionBias",
    "SwinV2Block",
    "MultiScaleDeformableAttentionV2",
    "ConvNeXtV2Block",
    "C3k2Block",
    "PSABlock",
]
