# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 回调基类与回调列表。
"""回调基类与回调列表。

所有输出统一走 ``trainer.logger``（utils.logging.TrainingLogger），不使用 print。
"""

import logging
from typing import Any, Dict, List, Optional


def _log(trainer: Any, level: str, msg: str) -> None:
    """输出到 trainer.logger；不存在时回退到标准 logging。"""
    logger = getattr(trainer, "logger", None)
    if logger is not None:
        getattr(logger, level, logger.info)(msg)
    else:
        getattr(logging.getLogger("dvisionix.callbacks"), level)(msg)


class Callback:
    """回调基类，可选择性地实现需要的钩子方法。"""

    def on_train_begin(self, trainer: Any) -> None:
        """训练开始时回调。"""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """训练结束时回调。"""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """每个 epoch 开始时回调。"""
        pass

    def on_validation_begin(self, trainer: Any) -> None:
        """验证开始前调用（EMA 等在此换权重）。"""
        pass

    def on_validation_end(self, trainer: Any) -> None:
        """验证结束后调用（EMA 等在此恢复权重）。"""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """每个 epoch 结束时回调（logs 为 epoch 指标）。"""
        pass

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str, batch=None) -> None:
        """每个 batch 开始时回调（mode: train/val）。"""
        pass

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        """每个 batch 结束时回调（logs 为 batch 指标）。"""
        pass

    # 状态持久化：默认无状态；有内部状态的子类应重写
    def state_dict(self) -> Dict[str, Any]:
        """返回回调状态字典（供断点续训）。"""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从状态字典恢复回调状态。"""
        return None


class CallbackList:
    """回调列表包装器，批量执行回调。"""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer: Any) -> None:
        """训练开始时批量回调。"""
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        """训练结束时批量回调。"""
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """每个 epoch 开始时批量回调。"""
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_validation_begin(self, trainer: Any) -> None:
        """验证开始时批量回调。"""
        for cb in self.callbacks:
            cb.on_validation_begin(trainer)

    def on_validation_end(self, trainer: Any) -> None:
        """验证结束时批量回调。"""
        for cb in self.callbacks:
            cb.on_validation_end(trainer)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """每个 epoch 结束时批量回调。"""
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str, batch=None) -> None:
        """每个 batch 开始时批量回调。"""
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch_idx, mode, batch)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        """每个 batch 结束时批量回调。"""
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, logs, mode, batch)

    def state_dict(self) -> Dict[str, Any]:
        """聚合所有回调的状态字典。"""
        result: Dict[str, Any] = {}
        for cb in self.callbacks:
            key = type(cb).__name__
            state = cb.state_dict()
            if not state:
                continue
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(state)
            else:
                result[key] = state
        return result

    def load_state_dict(self, states: Dict[str, Any]) -> None:
        """按类名分发恢复各回调状态。"""
        if not states:
            return
        buckets: Dict[str, list] = {}
        for cb in self.callbacks:
            buckets.setdefault(type(cb).__name__, []).append(cb)
        for key, entries in states.items():
            if key not in buckets:
                continue
            cbs = buckets[key]
            if isinstance(entries, list):
                for cb, entry in zip(cbs, entries):
                    cb.load_state_dict(entry)
            else:
                cbs[0].load_state_dict(entries)


__all__ = ["Callback", "CallbackList", "_log"]
