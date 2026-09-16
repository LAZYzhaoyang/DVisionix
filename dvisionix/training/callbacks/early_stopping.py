# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 早停回调。
"""早停回调。"""

import os
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
        """epoch 结束：监控指标连续未改善则触发早停。"""
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
                if self.restore_best_weights:
                    if self.best_weights is not None:
                        trainer.model.load_state_dict(self.best_weights)
                        _log(trainer, "info", "Restored best model weights")
                    elif not self._restore_from_disk(trainer):
                        # 续训后本进程没有内存快照：此前这里会**静默什么都不做**，
                        # 用户以为拿到了 best 模型，实际用的是最后一个 epoch 的权重。
                        _log(
                            trainer,
                            "warning",
                            "restore_best_weights 已开启，但本进程没有最优权重快照"
                            "（通常是 --resume 续训），且未能从 work_dir/checkpoints/best.pt "
                            "恢复；模型保持在最后一个 epoch 的权重。",
                        )

    def _restore_from_disk(self, trainer: Any) -> bool:
        """从 ``work_dir/checkpoints/best.pt`` 恢复最优权重。

        ``best_weights`` 是内存里的模型副本，**不会**写进 checkpoint（否则每次保存
        都要额外复制一份完整权重），所以续训后它恒为 ``None``；此时改用磁盘上的
        best.pt 兜底。
        """
        work_dir = getattr(trainer, "work_dir", None)
        model = getattr(trainer, "model", None)
        if not work_dir or model is None:
            return False
        path = os.path.join(work_dir, "checkpoints", "best.pt")
        if not os.path.exists(path):
            return False
        try:
            import torch

            # 该文件由本项目的 ModelCheckpoint 产出（含 RNG 等非张量状态，
            # 因此不能用 weights_only=True），只取其中的 model_state_dict。
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - 依赖具体文件损坏情况
            _log(trainer, "warning", f"读取最优 checkpoint 失败：{exc}")
            return False
        model_state = state.get("model_state_dict") if isinstance(state, dict) else state
        if not model_state:
            return False
        from ..checkpoint import strip_wrapper_prefixes

        try:
            model.load_state_dict(strip_wrapper_prefixes(model_state), strict=False)
        except Exception as exc:  # pragma: no cover
            _log(trainer, "warning", f"从最优 checkpoint 恢复权重失败：{exc}")
            return False
        _log(trainer, "info", f"Restored best model weights from {path}")
        return True

    def state_dict(self) -> Dict[str, Any]:
        """返回早停状态（best_value / patience 计数）。"""
        return {
            "best_value": self.best_value,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复早停状态。"""
        self.best_value = state.get("best_value", self.best_value)
        self.wait = state.get("wait", self.wait)
        self.stopped_epoch = state.get("stopped_epoch", self.stopped_epoch)


__all__ = ["EarlyStopping"]
