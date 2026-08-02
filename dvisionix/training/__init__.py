# D:\ZhaoyangProject\DVisionix\dvisionix\training\__init__.py

"""
训练模块（v0.3.1 目录重组）

- trainer.py：统一训练引擎（Task 组件驱动，支持 DDP / AMP / 梯度累积 / resume / work_dir）。
- tasks/：任务组件子包（BaseTask + 分类/检测/分割任务 + build_task）—— 主扩展点。
- callbacks/：回调子包（Callback/CallbackList + ProgressBar/ModelCheckpoint/EarlyStopping）—— 主扩展点。
- optim/：优化器/调度器注册表子包（OPTIMIZERS / SCHEDULERS + build_*）。
- workdir.py / builder.py / evaluation.py：工作目录隔离、配置驱动装配、检测评估。
- 注意：Loss 位于 dvisionix.models.losses（模型层组件）。

顶层 API 保持不变：from dvisionix.training import Trainer, ClassificationTask, build_task, ...
"""

from .trainer import Trainer
from .tasks import (
    BaseTask,
    ClassificationTask,
    DetectionTask,
    SegmentationTask,
    MultiLabelTask,
    build_task,
)
from .callbacks import (
    Callback,
    CallbackList,
    ProgressBar,
    ModelCheckpoint,
    EarlyStopping,
)
from .optim import OPTIMIZERS, build_optimizer, SCHEDULERS, build_scheduler
from .evaluation import evaluate_detection
from .workdir import (
    default_work_root,
    resolve_work_dir,
    find_latest_run,
    find_checkpoint,
    dump_config,
)
from .builder import build_callbacks, build_trainer

__all__ = [
    "Trainer",
    "BaseTask",
    "ClassificationTask",
    "DetectionTask",
    "SegmentationTask",
    "build_task",
    "Callback",
    "CallbackList",
    "ProgressBar",
    "ModelCheckpoint",
    "EarlyStopping",
    "OPTIMIZERS",
    "build_optimizer",
    "SCHEDULERS",
    "build_scheduler",
    "evaluate_detection",
    "default_work_root",
    "resolve_work_dir",
    "find_latest_run",
    "find_checkpoint",
    "dump_config",
    "build_callbacks",
    "build_trainer",
]