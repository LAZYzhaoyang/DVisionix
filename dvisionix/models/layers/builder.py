# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 归一化 / 激活层的按名称构建工具。
"""归一化 / 激活层的按名称构建工具。

统一以字符串（如 ``"bn"`` / ``"relu"``）或配置字典构建常用的 norm/激活层，
供自定义 layer 与 model 复用，避免到处 ``if name == ...`` 的分支。
"""

from typing import Any, Dict, Union

import torch.nn as nn

_NORM_LAYERS = {
    "bn": nn.BatchNorm2d,
    "bn1d": nn.BatchNorm1d,
    "batchnorm": nn.BatchNorm2d,
    "gn": nn.GroupNorm,
    "groupnorm": nn.GroupNorm,
    "in": nn.InstanceNorm2d,
    "instancenorm": nn.InstanceNorm2d,
    "ln": nn.LayerNorm,
    "layernorm": nn.LayerNorm,
}

_ACT_LAYERS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hardswish": nn.Hardswish,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def build_norm_layer(
    norm: Union[str, Dict[str, Any], nn.Module, None],
    num_features: int,
) -> nn.Module:
    """按名称/配置构建归一化层。

    Args:
        norm: 归一化类型。支持:
            - str: ``"bn"`` / ``"gn"`` / ``"in"`` / ``"ln"`` 等；``None`` 返回 Identity。
            - dict: ``{"type": "gn", "num_groups": 32}`` 形式，附带额外参数。
            - nn.Module: 直接返回（已实例化）。
        num_features: 通道数 / 特征维度。

    Returns:
        实例化后的归一化层。
    """
    if norm is None:
        return nn.Identity()
    if isinstance(norm, nn.Module):
        return norm

    if isinstance(norm, str):
        name, extra = norm, {}
    elif isinstance(norm, dict):
        cfg = dict(norm)
        name = cfg.pop("type", None) or cfg.pop("name", None)
        if name is None:
            raise KeyError("norm dict must contain a 'type' field")
        extra = cfg
    else:
        raise TypeError(f"Unsupported norm spec: {type(norm)}")

    key = name.lower()
    if key not in _NORM_LAYERS:
        raise KeyError(f"Unknown norm '{name}'. Available: {sorted(_NORM_LAYERS)}")
    layer_cls = _NORM_LAYERS[key]

    if layer_cls is nn.GroupNorm:
        num_groups = extra.pop("num_groups", 32)
        num_groups = min(num_groups, num_features)
        while num_features % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return layer_cls(num_groups, num_features, **extra)
    if layer_cls is nn.LayerNorm:
        return layer_cls(num_features, **extra)
    return layer_cls(num_features, **extra)


def build_activation_layer(
    act: Union[str, Dict[str, Any], nn.Module, None],
) -> nn.Module:
    """按名称/配置构建激活层。

    Args:
        act: 激活类型。支持 str / dict / nn.Module / None（None 返回 Identity）。

    Returns:
        实例化后的激活层。
    """
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    if isinstance(act, str):
        name, extra = act, {}
    elif isinstance(act, dict):
        cfg = dict(act)
        name = cfg.pop("type", None) or cfg.pop("name", None)
        if name is None:
            raise KeyError("act dict must contain a 'type' field")
        extra = cfg
    else:
        raise TypeError(f"Unsupported activation spec: {type(act)}")

    key = name.lower()
    if key not in _ACT_LAYERS:
        raise KeyError(f"Unknown activation '{name}'. Available: {sorted(_ACT_LAYERS)}")
    layer_cls = _ACT_LAYERS[key]
    # inplace 对支持的激活默认开启，减小显存
    if layer_cls in (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.SiLU) and "inplace" not in extra:
        extra["inplace"] = True
    return layer_cls(**extra)


__all__ = ["build_norm_layer", "build_activation_layer"]
