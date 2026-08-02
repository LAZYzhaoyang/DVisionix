# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型检查点保存回调（best.pt / last.pt / 可选 epoch 存档）。
"""模型检查点保存回调（best.pt / last.pt / 可选 epoch 存档）。"""

import os
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
        save_every_n_epochs: Optional[int] = None,
        max_epoch_checkpoints: Optional[int] = None,
    ):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename = filename  # 可选模板，如 "{epoch:03d}-{val_loss:.4f}.pt"
        self.save_every_n_epochs = save_every_n_epochs
        self.max_epoch_checkpoints = max_epoch_checkpoints
        self.epoch_checkpoints: list = []

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.is_better = (lambda x, best: x < best) if mode == "min" else (lambda x, best: x > best)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """epoch 结束：按监控指标保存 best.pt / last.pt / 可选存档。"""
        current = logs.get(self.monitor)

        if current is not None and self.is_better(current, self.best_value):
            self.best_value = current
            if self.save_best_only:
                trainer.save_checkpoint(str(self.save_dir / "best.pt"))
                _log(trainer, "info", f"Best checkpoint saved ({self.monitor}={current:.4f})")

        if self.save_last:
            trainer.save_checkpoint(str(self.save_dir / "last.pt"))

        if self.filename is not None:
            name = self.filename.format(
                epoch=epoch + 1, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
            )
            trainer.save_checkpoint(str(self.save_dir / name))

        if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
            self._save_epoch_checkpoint(trainer, epoch + 1)

    def _save_epoch_checkpoint(self, trainer: Any, epoch: int) -> None:
        name = f"epoch={epoch}.pt"
        path = str(self.save_dir / name)
        trainer.save_checkpoint(path)
        self.epoch_checkpoints.append(path)
        if self.max_epoch_checkpoints and len(self.epoch_checkpoints) > self.max_epoch_checkpoints:
            old = self.epoch_checkpoints.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass

    def state_dict(self) -> Dict[str, Any]:
        """返回 best_value 等检查点状态。"""
        return {"best_value": self.best_value}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复 best_value 等状态。"""
        self.best_value = state.get("best_value", self.best_value)


__all__ = ["ModelCheckpoint"]
