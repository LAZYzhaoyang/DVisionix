# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\training\\__init__.py

"""
训练模块

提供通用训练引擎、任务接口、回调系统和损失函数。

核心组件：
- Trainer: 通用训练引擎
- BaseTask: 任务基类
- ClassificationTask, DetectionTask, SegmentationTask: 内置标准任务
- Callback: 回调基类
- ModelCheckpoint, TensorBoardLogger, EarlyStopping: 内置回调
"""

from .trainer import Trainer
from .task import BaseTask, ClassificationTask, DetectionTask, SegmentationTask
from .callbacks import (
    Callback,
    CallbackList,
    ProgressBar,
    ModelCheckpoint,
    TensorBoardLogger,
    EarlyStopping,
    LearningRateScheduler,
)
from . import losses

__all__ = [
    "Trainer",
    "BaseTask",
    "ClassificationTask",
    "DetectionTask",
    "SegmentationTask",
    "Callback",
    "CallbackList",
    "ProgressBar",
    "ModelCheckpoint",
    "TensorBoardLogger",
    "EarlyStopping",
    "LearningRateScheduler",
    "losses",
]
