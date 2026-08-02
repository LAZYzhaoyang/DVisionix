# -*- coding: utf-8 -*-
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
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_validation_begin(self, trainer: Any) -> None:
        """验证开始前调用（EMA 等在此换权重）。"""
        pass

    def on_validation_end(self, trainer: Any) -> None:
        """验证结束后调用（EMA 等在此恢复权重）。"""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        pass

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str, batch=None) -> None:
        pass

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        pass

    # 状态持久化：默认无状态；有内部状态的子类应重写
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return None


class CallbackList:
    """回调列表包装器，批量执行回调。"""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_validation_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_validation_begin(trainer)

    def on_validation_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(trainer)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: Any, batch_idx: int, mode: str, batch=None) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch_idx, mode, batch)

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, logs, mode, batch)

    def state_dict(self) -> Dict[str, Any]:
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
