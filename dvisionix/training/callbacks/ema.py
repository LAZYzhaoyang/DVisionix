# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: EMA（指数滑动平均）回调。
"""EMA（指数滑动平均）回调。"""

import os
from typing import Any, Dict

import torch

from .base import Callback, _log


class EMA(Callback):
    """指数滑动平均：维护影子权重，验证时换入 EMA 权重，结束后恢复。

    Args:
        decay: 滑动系数（0.999 常用）。
        swap_for_validation: 验证时是否使用 EMA 权重。
    """

    def __init__(
        self,
        decay: float = 0.999,
        swap_for_validation: bool = True,
        decay_warmup_epochs: int = 0,
        save_final: bool = False,
    ):
        self.decay = float(decay)
        self.swap_for_validation = swap_for_validation
        self.decay_warmup_epochs = int(decay_warmup_epochs)
        self.save_final = bool(save_final)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._saved: Dict[str, torch.Tensor] = {}

    def _effective_decay(self, trainer: Any) -> float:
        """decay 调度：warmup 期间从 0.5 线性升到目标 decay，之后恒定。"""
        if self.decay_warmup_epochs <= 0:
            return self.decay
        progress = (trainer.current_epoch + 1) / self.decay_warmup_epochs
        progress = min(1.0, max(0.0, progress))
        return 0.5 + (self.decay - 0.5) * progress

    def on_train_begin(self, trainer: Any) -> None:
        """初始化 EMA 影子参数。"""
        state = trainer.model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        """每个训练 batch 后更新 EMA 影子权重。"""
        if mode != "train":
            return
        with torch.no_grad():
            decay = self._effective_decay(trainer)
            for k, v in trainer.model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k] = decay * self.shadow[k] + (1.0 - decay) * v.float()

    def on_validation_begin(self, trainer: Any) -> None:
        """验证前把 EMA 权重同步到模型。"""
        if not self.swap_for_validation:
            return
        self._saved = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        state = {k: v.float() for k, v in self.shadow.items()}
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        if missing:
            _log(trainer, "warning", f"EMA swap missing keys: {missing}")

    def on_validation_end(self, trainer: Any) -> None:
        """验证后恢复模型原始权重。"""
        if not self.swap_for_validation:
            return
        trainer.model.load_state_dict(self._saved)

    def on_train_end(self, trainer: Any) -> None:
        """训练结束：按 save_final 导出 ema_last.pt。"""
        if not self.save_final or not getattr(trainer, "work_dir", None):
            return
        path = os.path.join(trainer.work_dir, "ema_last.pt")
        torch.save({k: v.float() for k, v in self.shadow.items()}, path)
        _log(trainer, "info", f"EMA 权重已导出到 {path}")

    def state_dict(self) -> Dict[str, Any]:
        """返回 EMA 影子权重状态。"""
        return {
            "decay": self.decay,
            "decay_warmup_epochs": self.decay_warmup_epochs,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复 EMA 影子权重。"""
        self.decay = state.get("decay", self.decay)
        self.decay_warmup_epochs = state.get("decay_warmup_epochs", self.decay_warmup_epochs)
        self.shadow = state.get("shadow", self.shadow)


__all__ = ["EMA"]
