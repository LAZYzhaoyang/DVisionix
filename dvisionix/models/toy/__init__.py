# -*- coding: utf-8 -*-
"""教学级模型（toy）：SimpleCNN / SimpleSegmentationModel / GridDetectionModel。

这些模型仅用于演示与快速验证数据/训练流程，不与组件化模型混放；
生产请使用组件化模型（LinearClassifier / SegmentationModel / FCOSDetector / RetinaNetDetector）。
"""

from ...registry import MODELS
from .classification import SimpleCNN
from .detection import GridDetectionModel
from .segmentation import SimpleSegmentationModel

_MODELS = (SimpleCNN, SimpleSegmentationModel, GridDetectionModel)
_ALIASES = {
    SimpleCNN: "simple_cnn",
    SimpleSegmentationModel: "simple_segmentation",
    GridDetectionModel: "grid_detection",
}
for _cls in _MODELS:
    if _cls.__name__ not in MODELS:
        MODELS.register(_cls)
for _cls, _alias in _ALIASES.items():
    if _alias not in MODELS:
        MODELS.register(_cls, name=_alias)

__all__ = ["SimpleCNN", "SimpleSegmentationModel", "GridDetectionModel"]
