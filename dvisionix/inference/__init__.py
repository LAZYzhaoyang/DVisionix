# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 推理子包：预处理、统一结果对象与预测器。
"""推理子包：预处理、统一结果对象与预测器。

导出三部分：

- 预处理：:class:`PreprocessSpec` / :class:`PreprocessMeta` / :class:`PreprocessPipeline`；
- 结果：:class:`Results`；
- 预测器：:class:`BasePredictor` 及其任务子类、:func:`build_predictor`。
"""

from .predictors import (
    BasePredictor,
    ClassificationPredictor,
    DetectionPredictor,
    SegmentationPredictor,
    build_predictor,
)
from .preprocess import PreprocessMeta, PreprocessPipeline, PreprocessSpec
from .results import Results

__all__ = [
    "PreprocessSpec",
    "PreprocessMeta",
    "PreprocessPipeline",
    "Results",
    "BasePredictor",
    "ClassificationPredictor",
    "DetectionPredictor",
    "SegmentationPredictor",
    "build_predictor",
]
