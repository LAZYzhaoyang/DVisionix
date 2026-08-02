# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 数据模块（统一接口 + 主流公开数据集 + 原子变换 + 自由组合）。
"""数据模块（统一接口 + 主流公开数据集 + 原子变换 + 自由组合）。

- ``Sample`` 契约（``data.sample``）：所有 dataset / transform / collate 共用的字段名约定。
- ``BaseDataset``（``data.base``）：``__getitem__`` 流程统一为 sample -> transforms -> dict。
- ``transforms``：原子 image / 几何同步 / 标签 / 第三方封装 + ``TransformPipeline``。
- ``presets``：主流公开数据集（CIFAR / ImageNet / COCO / VOC / Cityscapes / ADE20K / ImageFolder）。
- ``CustomDataset``：用户最简自定义模板。
- ``collate``：检测 / 分割任务的变长 collate。
"""

from . import (
    presets,  # 注册到 DATASETS
    transforms,
)
from .base import BaseDataset
from .collate import detection_collate, segmentation_collate
from .datasets.custom import CustomDataset
from .sample import ImageInfo, ImageMode, NormalizationSpec, Sample
from .transforms import (  # noqa: E402  预设 pipeline 别名
    # 第三方
    AlbumentationsWrapper,
    BaseTransform,
    BoxesToTensor,
    BoxSyncPad,
    BoxSyncRandomCrop,
    BoxSyncRandomHorizontalFlip,
    # 几何同步
    BoxSyncResize,
    CenterCrop,
    ClassificationTransforms,
    ColorJitter,
    DetectionTransforms,
    ImageNormalize,
    # 原子 image
    ImageResize,
    # 标签
    LabelToTensor,
    MaskToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    SegmentationTransforms,
    ToTensor,
    TransformPipeline,
    build_pipeline,
    build_transform,
)

__all__ = [
    # 样本契约
    "Sample",
    "ImageInfo",
    "ImageMode",
    "NormalizationSpec",
    # 数据集
    "BaseDataset",
    "CustomDataset",
    "transforms",
    "presets",
    # collate
    "detection_collate",
    "segmentation_collate",
    # 变换基础
    "BaseTransform",
    "TransformPipeline",
    "build_transform",
    "build_pipeline",
    # 原子 image
    "ImageResize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomCrop",
    "CenterCrop",
    "ColorJitter",
    "ImageNormalize",
    "ToTensor",
    # 几何同步
    "BoxSyncResize",
    "BoxSyncRandomHorizontalFlip",
    "BoxSyncRandomCrop",
    "BoxSyncPad",
    # 标签
    "LabelToTensor",
    "BoxesToTensor",
    "MaskToTensor",
    # 第三方
    "AlbumentationsWrapper",
    # 预设 pipeline 别名（兼容旧 API）
    "ClassificationTransforms",
    "DetectionTransforms",
    "SegmentationTransforms",
]


# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict

from ..registry import DATASETS


def build_dataset(cfg: Dict[str, Any]):
    """从配置字典构建数据集（任意已注册数据集类型）。"""
    return DATASETS.build(dict(cfg))


__all__ = __all__ + ["DATASETS", "build_dataset"]
