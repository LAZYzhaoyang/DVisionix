# D:\ZhaoyangProject\DVisionix\dvisionix\__init__.py

"""
DVisionix: 深度学习算法库

一个模块化、可扩展的深度学习算法库，支持分类、检测、分割等多种任务。

核心特性：
- 统一的数据接口，支持所有任务
- 通用的训练引擎，支持自定义任务逻辑（Task 组件）
- 丰富的回调系统，支持灵活的训练控制
- Loss 作为模型层组件（models.losses），可继承、可自由组合
- 完整的指标计算，涵盖所有常见任务
- 多卡训练（DDP）、工作目录隔离、自动断点续训

快速开始（配置驱动）：
    python tools/train.py --config configs/classification/demo_synthetic.yaml

编程接口：
    from dvisionix.models import build_model
    from dvisionix.training import build_task, Trainer
"""

__version__ = "0.4.0"

from . import data
from . import models
from . import training
from . import metrics
from . import utils
from . import config
from . import export

__all__ = [
    "data",
    "models",
    "training",
    "metrics",
    "utils",
    "config",
    "export",
    "__version__",
]


# =============================================================================
# 注册表与构建入口（配置驱动）
# =============================================================================
from .registry import (
    Registry, build_from_cfg, MODELS, BACKBONES, NECKS, HEADS,
    DATASETS, TRANSFORMS, TASKS, LOSSES, METRICS,
)
from .models import build_model
from .models.losses import build_loss
from .training import build_task
from .metrics import build_metric
from .data import build_dataset

__all__ = __all__ + [
    "Registry", "build_from_cfg", "MODELS", "BACKBONES", "NECKS", "HEADS",
    "DATASETS", "TRANSFORMS", "TASKS", "LOSSES", "METRICS",
    "build_model", "build_task", "build_loss", "build_metric", "build_dataset",
]