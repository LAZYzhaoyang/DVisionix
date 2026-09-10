# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 数据变换模块（原子 + 几何同步 + 第三方封装）。
"""数据变换模块（原子 + 几何同步 + 第三方封装）。

设计：
- 原子变换（``image.py``）：只动 ``image`` 字段，任务无关。
- 几何同步（``geometric.py``）：同时更新 ``image`` / ``boxes`` / ``mask``，
  分类 / 检测 / 分割复用同一套。
- 标签（``labels.py``）：numpy -> torch.Tensor。
- 第三方（``third_party.py``）：把 albumentations / kornia / torchvision.transforms.v2
  等库适配为 ``BaseTransform`` 协议。
- ``base.py``：``BaseTransform`` / ``TransformPipeline``。
- ``builder.py``：``build_transform`` / ``build_pipeline``，支持实例 / 配置字典 / 字符串混用。
"""

from ...registry import TRANSFORMS
from .base import BaseTransform, TransformPipeline
from .builder import build_pipeline, build_transform
from .geometric import (
    BoxSyncPad,
    BoxSyncRandomCrop,
    BoxSyncRandomHorizontalFlip,
    BoxSyncResize,
)
from .image import (
    CenterCrop,
    ColorJitter,
    ImageNormalize,
    ImageResize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
from .labels import BoxesToTensor, LabelToTensor, MaskToTensor
from .simclr import SimCLRTransforms
from .third_party import AlbumentationsWrapper

__all__ = [
    # 基础
    "BaseTransform",
    "TransformPipeline",
    "build_transform",
    "build_pipeline",
    # 图像原子变换
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
    # 第三方封装
    "AlbumentationsWrapper",
    # SimCLR
    "SimCLRTransforms",
]


def _build_classification_pipeline(image_size, mean, std, train):
    if train:
        return [
            ImageResize((int(image_size * 1.1), int(image_size * 1.1))),
            RandomCrop((image_size, image_size)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ImageNormalize(mean, std),
        ]
    return [
        ImageResize((image_size, image_size)),
        CenterCrop((image_size, image_size)),
        ImageNormalize(mean, std),
    ]


@TRANSFORMS.register()
@TRANSFORMS.register(name="classification_transforms")
class ClassificationTransforms(TransformPipeline):
    """分类预设 pipeline（train=True/False）—— 兼容旧 API。"""

    task_type = "classification"

    def __init__(
        self,
        train: bool = True,
        image_size: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__(
            _build_classification_pipeline(image_size, mean, std, train)
            + [ToTensor(keys=("image",)), LabelToTensor()]
        )
        self.train = train
        self.image_size = image_size


@TRANSFORMS.register()
@TRANSFORMS.register(name="detection_transforms")
class DetectionTransforms(TransformPipeline):
    """检测预设 pipeline。"""

    task_type = "detection"

    def __init__(
        self,
        train: bool = True,
        image_size: int = 640,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        # 训练时先放大再随机裁剪：直接把图 resize 到目标尺寸再裁剪会退化成恒等操作
        # （随机偏移恒为 0），数据增强完全失效（CodePlan 7.1.2 D9）。
        # 1.1x 与分类预置管线 _build_classification_pipeline 保持一致。
        resize_to = (
            (int(image_size * 1.1), int(image_size * 1.1)) if train else (image_size, image_size)
        )
        ops = [BoxSyncResize(resize_to)]
        if train:
            ops += [BoxSyncRandomHorizontalFlip(p=0.5), BoxSyncRandomCrop((image_size, image_size))]
        ops += [ImageNormalize(mean, std), ToTensor(keys=("image",)), BoxesToTensor()]
        super().__init__(ops)
        self.train = train
        self.image_size = image_size


@TRANSFORMS.register()
@TRANSFORMS.register(name="segmentation_transforms")
class SegmentationTransforms(TransformPipeline):
    """分割预设 pipeline。"""

    task_type = "segmentation"

    def __init__(
        self,
        train: bool = True,
        image_size: int = 512,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        ops = [BoxSyncResize((image_size, image_size))]
        if train:
            ops += [BoxSyncRandomHorizontalFlip(p=0.5), ColorJitter(brightness=0.2)]
        ops += [ImageNormalize(mean, std), ToTensor(keys=("image",)), MaskToTensor()]
        super().__init__(ops)
        self.train = train
        self.image_size = image_size


__all__ = __all__ + [
    "ClassificationTransforms",
    "DetectionTransforms",
    "SegmentationTransforms",
]
