# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型基类与任务类型契约。
"""模型基类与任务类型契约。

教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）已迁移到
``dvisionix.models.toy``，本文件只保留 ``BaseModel`` 契约。
"""

from typing import Optional

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
        - 可选实现 ``init_weights`` 做权重初始化，``from_config`` 支持配置化构建。

    提供 freeze/unfreeze、参数统计、设备查询等通用能力。
    """

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
        """获取模型所在的设备"""
        return next(self.parameters()).device


__all__ = ["BaseModel", "TASK_TYPES"]
