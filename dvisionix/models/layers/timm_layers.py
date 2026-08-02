# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 封装来自 timm 的层（timm.layers）。
"""封装来自 timm 的层（timm.layers）。

timm 提供了大量高质量的即用层（DropPath / SqueezeExcite / Mlp / ConvNormAct 等），
本模块将它们以统一方式暴露给 DVisionix：

- ``create_timm_layer(name, **kwargs)``: 按名称实例化 timm.layers 中的任意层/工厂函数。
- ``list_timm_layers()``: 列出 timm.layers 中可用的层名称。
- 部分常用 timm 层会在导入时注册到全局 ``LAYERS`` 注册表（带 ``timm_`` 前缀，避免与内置层重名）。
"""

from typing import Any, List

from ...registry import LAYERS


def _require_timm_layers():
    try:
        import timm.layers as timm_layers

        return timm_layers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "timm is not installed. Please install it with: pip install timm"
        ) from exc


def create_timm_layer(name: str, *args: Any, **kwargs: Any):
    """按名称实例化 timm.layers 中的层或调用其工厂函数。

        Args:
            name: timm.layers 中的类名或工厂函数名，如 ``"SqueezeExcite"``、
                ``"DropPath"``、``"create_conv2d"``、``"get_act_layer"``。
            *args, **kwargs: 传给该类/函数的参数。

        Returns:
            实例化后的 `
    n.Module`` 或工厂函数的返回值。

        Examples:
            >>> se = create_timm_layer("SqueezeExcite", 64, rd_ratio=0.25)
            >>> dp = create_timm_layer("DropPath", drop_prob=0.1)
            >>> conv = create_timm_layer("create_conv2d", 32, 64, kernel_size=3)
    """
    timm_layers = _require_timm_layers()
    if not hasattr(timm_layers, name):
        raise AttributeError(
            f"timm.layers has no attribute '{name}'. "
            f"Use list_timm_layers() to inspect available names."
        )
    target = getattr(timm_layers, name)
    if not callable(target):
        raise TypeError(f"timm.layers.{name} is not callable")
    return target(*args, **kwargs)


def list_timm_layers() -> List[str]:
    """列出 timm.layers 中可用的（首字母大写的类）层名称。"""
    timm_layers = _require_timm_layers()
    names = [n for n in dir(timm_layers) if not n.startswith("_")]
    return sorted(names)


class _TimmLayerFactory:
    """把 ``create_timm_layer`` 适配为可注册到 LAYERS 的可调用工厂。

    使得配置 ``{"type": "timm_squeeze_excite", "channels": 64}`` 也能构建。
    """

    def __init__(self, timm_name: str) -> None:
        self._timm_name = timm_name

    def __call__(self, **kwargs: Any):
        return create_timm_layer(self._timm_name, **kwargs)


# 将部分常用 timm 层以 timm_ 前缀注册（延迟到调用时才 import timm，避免导入期强依赖）
_TIMM_LAYER_ALIASES = {
    "timm_squeeze_excite": "SqueezeExcite",
    "timm_drop_path": "DropPath",
    "timm_mlp": "Mlp",
    "timm_conv_norm_act": "ConvNormAct",
}

for _alias, _timm_name in _TIMM_LAYER_ALIASES.items():
    if _alias not in LAYERS:
        LAYERS.register(_TimmLayerFactory(_timm_name), name=_alias)


__all__ = ["create_timm_layer", "list_timm_layers"]
