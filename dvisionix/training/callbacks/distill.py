# -*- coding: utf-8 -*-
"""知识蒸馏回调（DistillCallback）。

把 teacher 模型挂在 trainer 上，每个训练 batch 计算 teacher logits（no_grad）并存入
``trainer.teacher_logits``，供自定义任务在 training_step 中结合
``dvisionix.models.losses.DistillationLoss`` 使用。
"""

from typing import Any, Dict

import torch

from .base import Callback


class DistillCallback(Callback):
    """知识蒸馏回调：管理 teacher 模型并产出 soft targets。

    Args:
        teacher: teacher 模型（forward 输出 logits）。
        temperature: 蒸馏温度（用于 softmax）。
    """

    def __init__(self, teacher, temperature: float = 4.0):
        self.teacher = teacher
        self.temperature = float(temperature)
        self._batch_idx = 0

    def on_train_begin(self, trainer: Any) -> None:
        self.teacher = self.teacher.to(trainer.device).eval()
        trainer.teacher_logits = None
        trainer.distill_temperature = self.temperature

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str) -> None:
        if mode != "train":
            return
        images = batch["image"].to(trainer.device)
        with torch.no_grad():
            teacher_out = self.teacher(images)
        if isinstance(teacher_out, dict):
            teacher_out = teacher_out.get("logits", teacher_out)
        trainer.teacher_logits = teacher_out

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        trainer.teacher_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {"teacher": self.teacher.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "teacher" in state:
            self.teacher.load_state_dict(state["teacher"])


__all__ = ["DistillCallback"]