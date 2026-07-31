# -*- coding: utf-8 -*-
"""变换的构建工具。

支持从配置字典 / 字符串 / BaseTransform 实例构建原子变换或组合管道。
"""

from typing import Any, Dict, Union

from .base import BaseTransform, TransformPipeline
from ...registry import TRANSFORMS


TransformSpec = Union[BaseTransform, TransformPipeline, Dict[str, Any], str]


def build_transform(spec: TransformSpec) -> BaseTransform:
    """从实例 / 配置字典 / 字符串构建单个原子变换。

    Args:
        spec: BaseTransform 实例、``{"type": "image_resize", ...}`` 字典，
            或注册名字符串（如 ``"image_resize"``，将按无参构建）。
    """
    if isinstance(spec, (BaseTransform, TransformPipeline)):
        return spec
    if isinstance(spec, str):
        return TRANSFORMS.build({"type": spec})
    if isinstance(spec, dict):
        return TRANSFORMS.build(dict(spec))
    raise TypeError(f"Unsupported transform spec: {type(spec)}")


def build_pipeline(specs) -> TransformPipeline:
    """从 spec 列表构建一个组合管道。"""
    return TransformPipeline(specs)


__all__ = ["build_transform", "build_pipeline"]