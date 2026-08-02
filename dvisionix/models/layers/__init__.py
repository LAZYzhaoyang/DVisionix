# -*- coding: utf-8 -*-
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
from .attention import PositionEmbeddingSine
from .basic import MLP, ConvNormAct, DropPath, SEBlock
from .builder import build_activation_layer, build_norm_layer
from .convnext import ConvNeXtBlock
from .csp import CSPLayer
from .deformable_attention import MultiScaleDeformableAttention
from .elan import EELANLayer, ELANLayer
from .mbconv import MBConvBlock
from .reversible import ReversibleBlock
from .timm_layers import create_timm_layer, list_timm_layers


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
__all__ = __all__ + ["PositionEmbeddingSine"]
