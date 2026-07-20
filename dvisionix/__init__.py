# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\__init__.py

"""
DVisionix: 深度学习算法库

一个模块化、可扩展的深度学习算法库，支持分类、检测、分割等多种任务。

核心特性：
- 统一的数据接口，支持所有任务
- 通用的训练引擎，支持自定义任务逻辑
- 丰富的回调系统，支持灵活的训练控制
- 完整的指标计算，涵盖所有常见任务

快速开始：
    from dvisionix.data import DatasetFactory
    from dvisionix.models import SimpleCNN
    from dvisionix.training import Trainer, ClassificationTask, ModelCheckpoint

    # 创建数据集
    train_dataset = DatasetFactory.create("cifar10", root="./data", train=True)

    # 创建模型
    model = SimpleCNN(num_classes=10)

    # 定义任务和回调
    task = ClassificationTask(num_classes=10)
    callbacks = [ModelCheckpoint(save_dir="./checkpoints")]

    # 创建训练器并开始训练
    trainer = Trainer(task, train_loader, callbacks=callbacks)
    trainer.fit(model)
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import training
from . import metrics
from . import utils

__all__ = [
    "data",
    "models",
    "training",
    "metrics",
    "utils",
    "__version__",
]
