# -*- coding: utf-8 -*-
"""任务组件子包：BaseTask + 内置任务 + build_task（配置驱动）。

新增任务：在 tasks/ 下新建一个文件（继承 BaseTask），在顶部 ``@TASKS.register()`` 注册即可。
"""

from typing import Any, Dict

from ...registry import TASKS
from .base import BaseTask
from .classification import ClassificationTask
from .detection import DetectionTask
from .maskformer import MaskFormerTask
from .multi_label import MultiLabelTask
from .segmentation import SegmentationTask
from .simclr import SimCLRTask

for _cls in (
    ClassificationTask,
    DetectionTask,
    SegmentationTask,
    MultiLabelTask,
    MaskFormerTask,
    SimCLRTask,
):
    if _cls.__name__ not in TASKS:
        TASKS.register(_cls)


def build_task(cfg: Dict[str, Any]):
    """从配置构建任务实例。

    例如::

        build_task({"type": "ClassificationTask", "num_classes": 10,
                    "optimizer_cfg": {"type": "adamw", "lr": 1e-3},
                    "loss": {"type": "focal", "gamma": 2.0}})
    """
    return TASKS.build(dict(cfg))


__all__ = [
    "BaseTask",
    "ClassificationTask",
    "DetectionTask",
    "SegmentationTask",
    "MultiLabelTask",
    "MaskFormerTask",
    "SimCLRTask",
    "build_task",
]
