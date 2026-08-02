# -*- coding: utf-8 -*-
"""EMA（指数滑动平均）回调。"""

from typing import Any, Dict

import torch

from .base import Callback, _log


class EMA(Callback):
    """指数滑动平均：维护影子权重，验证时换入 EMA 权重，结束后恢复。

    Args:
        decay: 滑动系数（0.999 常用）。
        swap_for_validation: 验证时是否使用 EMA 权重。
    """

    def __init__(self, decay: float = 0.999, swap_for_validation: bool = True):
        self.decay = float(decay)
        self.swap_for_validation = swap_for_validation
        self.shadow: Dict[str, torch.Tensor] = {}
        self._saved: Dict[str, torch.Tensor] = {}

    def on_train_begin(self, trainer: Any) -> None:
        state = trainer.model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        if mode != "train":
            return
        with torch.no_grad():
            for k, v in trainer.model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def on_validation_begin(self, trainer: Any) -> None:
        if not self.swap_for_validation:
            return
        self._saved = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        state = {k: v.float() for k, v in self.shadow.items()}
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        if missing:
            _log(trainer, "warning", f"EMA swap missing keys: {missing}")

    def on_validation_end(self, trainer: Any) -> None:
        if not self.swap_for_validation:
            return
        trainer.model.load_state_dict(self._saved)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", self.shadow)


__all__ = ["EMA"]
