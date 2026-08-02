# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 知识蒸馏回调（DistillCallback）。
"""知识蒸馏回调（DistillCallback）。

把 teacher 模型挂在 trainer 上，每个训练 batch 计算 teacher logits（no_grad）并存入
``trainer.teacher_logits``，供自定义任务在 training_step 中结合
``dvisionix.models.losses.DistillationLoss`` 使用。
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .base import Callback


class DistillCallback(Callback):
    """知识蒸馏回调：管理 teacher 模型并产出 soft targets。

    Args:
        teacher: teacher 模型（forward 输出 logits）。
        temperature: 蒸馏温度（用于 softmax）。
    """

    def __init__(
        self,
        teacher,
        temperature: float = 4.0,
        feature_extractor: Optional[Callable[[nn.Module, Any], List[Any]]] = None,
    ):
        self.teacher = teacher
        self.temperature = float(temperature)
        self.feature_extractor = feature_extractor
        self._batch_idx = 0

    def on_train_begin(self, trainer: Any) -> None:
        """将 teacher 置为 eval 并初始化缓存字段。"""
        self.teacher = self.teacher.to(trainer.device).eval()
        trainer.teacher_logits = None
        trainer.teacher_features = None
        trainer.distill_temperature = self.temperature

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str, batch=None) -> None:
        """每个训练 batch 计算 teacher logits / 中间特征。"""
        if mode != "train" or batch is None:
            return
        images = batch["image"].to(trainer.device)
        with torch.no_grad():
            teacher_out = self.teacher(images)
        if isinstance(teacher_out, dict):
            teacher_out = teacher_out.get("logits", teacher_out)
        trainer.teacher_logits = teacher_out
        if self.feature_extractor is not None:
            trainer.teacher_features = self.feature_extractor(self.teacher, images)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """epoch 结束清空 teacher 缓存。"""
        trainer.teacher_logits = None
        trainer.teacher_features = None

    def state_dict(self) -> Dict[str, Any]:
        """返回 teacher 状态字典。"""
        return {"teacher": self.teacher.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复 teacher 权重。"""
        if "teacher" in state:
            self.teacher.load_state_dict(state["teacher"])


__all__ = ["DistillCallback"]
