# -*- coding: utf-8 -*-
"""模型检查点保存回调（best.pt / last.pt / 可选 epoch 存档）。"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import Callback, _log


class ModelCheckpoint(Callback):
    """模型检查点保存（best.pt / last.pt / 可选 epoch 存档）。"""

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        filename: Optional[str] = None,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename = filename  # 可选模板，如 "{epoch:03d}-{val_loss:.4f}.pt"

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.is_better = (lambda x, best: x < best) if mode == "min" else (lambda x, best: x > best)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)

        if current is not None and self.is_better(current, self.best_value):
            self.best_value = current
            if self.save_best_only:
                trainer.save_checkpoint(str(self.save_dir / "best.pt"))
                _log(trainer, "info", f"Best checkpoint saved ({self.monitor}={current:.4f})")

        if self.save_last:
            trainer.save_checkpoint(str(self.save_dir / "last.pt"))

        if self.filename is not None:
            name = self.filename.format(epoch=epoch + 1, **{k: v for k, v in logs.items() if isinstance(v, (int, float))})
            trainer.save_checkpoint(str(self.save_dir / name))

    def state_dict(self) -> Dict[str, Any]:
        return {"best_value": self.best_value}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_value = state.get("best_value", self.best_value)


__all__ = ["ModelCheckpoint"]