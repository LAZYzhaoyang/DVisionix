# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型基类与任务类型契约。
"""模型基类与任务类型契约。

教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）已迁移到
``dvisionix.models.toy``，本文件只保留 ``BaseModel`` 契约。
"""

from typing import ClassVar, Optional

import torch
import torch.nn as nn

# 合法任务类型集合，用于校验 BaseModel.task_type
TASK_TYPES = {"classification", "detection", "segmentation"}


class BaseModel(nn.Module):
    """所有模型的基类。

    契约:
        - ``forward`` 只返回 **原始预测**（logits / raw 张量 / dict），不包含 NMS / decode
          等后处理；后处理统一放在 ``dvisionix.models.postprocess`` 或独立解码器中。
        - ``task_type`` 应取 ``TASK_TYPES`` 之一。
        - ``supported_tasks`` 可声明模型支持的任务能力集合（P2-1）；默认空元组表示
          "未显式声明，由 ``task_type`` 推导"。
        - 可选实现 ``init_weights`` 做权重初始化，``from_config`` 支持配置化构建。

    提供 freeze/unfreeze、参数统计、设备查询等通用能力。
    """

    #: 模型支持的任务能力集合（如 ``("classification",)``）。
    #: 默认空元组表示"由 ``task_type`` 推导"；多任务模型可显式声明多个能力。
    supported_tasks: ClassVar[tuple[str, ...]] = ()

    def __init__(self, task_type: Optional[str] = None):
        super().__init__()
        if task_type is not None and task_type not in TASK_TYPES:
            raise ValueError(
                f"Invalid task_type {task_type!r}, expected one of {sorted(TASK_TYPES)}"
            )
        self.task_type = task_type

    def forward(self, x, **kwargs):
        """前向传播，子类必须实现，只返回原始预测。"""
        raise NotImplementedError

    def init_weights(self):
        """权重初始化钩子，子类可覆盖。默认不做额外处理。"""
        return None

    @classmethod
    def from_config(cls, cfg):
        """从配置字典构建模型。子类可覆盖以定制构建逻辑。"""
        return cls(**dict(cfg))

    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def get_device(self) -> torch.device:
        """获取模型所在的设备。

        无参数模型（例如只由 buffer 或纯函数构成的模型）不会让 ``next(...)`` 抛
        ``StopIteration``：先看参数、再看 buffer，都没有则返回 CPU。
        """
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")


def model_task_types(model: nn.Module) -> tuple[str, ...]:
    """返回模型声明的任务能力。

    取值优先级（契约 P2-1）：

    1. ``model.supported_tasks``（实例或类属性）非空 → 直接使用；
    2. 实例/类的 ``task_type`` 非空 → 包装成单元素元组；
    3. 两者都没有 → 空元组 ``()``。

    空能力表示"模型没有声明任何完整任务能力"（例如 ``FeatureBackboneBase`` 这类
    非任务模型），由调用方（``training.contract``）决定如何报错。

    Args:
        model: 待查询的模型（``nn.Module`` 即可，不要求继承 ``BaseModel``）。

    Returns:
        模型声明的任务能力元组；未声明时为空元组。
    """
    supported = getattr(model, "supported_tasks", ())
    if isinstance(supported, str):
        # 容错：误写成字符串时按单个能力处理，避免 tuple("classification") 拆成字符
        supported = (supported,)
    if supported:
        return tuple(str(item) for item in supported)

    task_type = getattr(model, "task_type", None)
    if task_type:
        return (str(task_type),)
    return ()


__all__ = ["BaseModel", "TASK_TYPES", "model_task_types"]
