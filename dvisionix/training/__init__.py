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
from .evaluation import evaluate_detection

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
    "evaluate_detection",
]


# =============================================================================
# 注册表集成（配置驱动构建）
# =============================================================================
from typing import Any, Dict
from ..registry import TASKS

for _cls in (ClassificationTask, DetectionTask, SegmentationTask):
    if _cls.__name__ not in TASKS:
        TASKS.register(_cls)


def build_task(cfg: Dict[str, Any]):
    """从配置构建任务实例。

    例如::

        build_task({"type": "ClassificationTask", "num_classes": 10, "learning_rate": 1e-3})
    """
    return TASKS.build(dict(cfg))


__all__ = __all__ + ["TASKS", "build_task"]
