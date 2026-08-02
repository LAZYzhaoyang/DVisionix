# -*- coding: utf-8 -*-
"""训练进度显示回调。"""

import time
from typing import Any, Dict

from .base import Callback, _log


class ProgressBar(Callback):
    """训练进度显示（通过 trainer.logger 输出）。"""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        self.epoch_start_time = time.time()
        _log(trainer, "info", f"Epoch [{epoch + 1}/{trainer.max_epochs}]")

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        if batch_idx % self.log_interval == 0:
            prefix = "train" if mode == "train" else "val"
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            _log(trainer, "info", f"  [{prefix}] batch {batch_idx}: {metrics_str}")

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        epoch_time = time.time() - self.epoch_start_time
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        _log(
            trainer, "info", f"Epoch [{epoch + 1}] summary: {metrics_str} | time: {epoch_time:.1f}s"
        )


__all__ = ["ProgressBar"]
