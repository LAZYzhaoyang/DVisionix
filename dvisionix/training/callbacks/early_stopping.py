# -*- coding: utf-8 -*-
"""早停回调。"""

from typing import Any, Dict

from .base import Callback, _log


class EarlyStopping(Callback):
    """早停机制。"""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        if mode == "min":
            self.best_value = float("inf")
            self.is_better = lambda x, best: x < best - min_delta
        else:
            self.best_value = float("-inf")
            self.is_better = lambda x, best: x > best + min_delta

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.is_better(current, self.best_value):
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                _log(trainer, "warning", f"Early stopping at epoch {epoch + 1}")
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                    _log(trainer, "info", "Restored best model weights")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_value": self.best_value,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_value = state.get("best_value", self.best_value)
        self.wait = state.get("wait", self.wait)
        self.stopped_epoch = state.get("stopped_epoch", self.stopped_epoch)


__all__ = ["EarlyStopping"]
