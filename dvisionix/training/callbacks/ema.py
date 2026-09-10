# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: EMA（指数滑动平均）回调。
"""EMA（指数滑动平均）回调。"""

import os
from typing import Any, Dict, List

import torch

from .base import Callback, _log


class EMA(Callback):
    """指数滑动平均：维护影子权重，验证时换入 EMA 权重，结束后恢复。

    只为**浮点**参数与浮点 buffer 维护影子；整型 buffer（如
    ``num_batches_tracked``）是计数器，不参与滑动平均。

    Args:
        decay: 滑动系数（0.999 常用）。
        swap_for_validation: 验证时是否使用 EMA 权重。
        decay_warmup_epochs: decay 从 0.5 线性升到目标值的 warmup 轮数（0 表示不 warmup）。
        save_final: 训练结束后是否导出 ``ema_last.pt``。
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
        self._keys: List[str] = []
        self._pairs: List[tuple] = []
        self._saved: Dict[str, torch.Tensor] = {}

    def _effective_decay(self, trainer: Any) -> float:
        """decay 调度：warmup 期间从 0.5 线性升到目标 decay，之后恒定。"""
        if self.decay_warmup_epochs <= 0:
            return self.decay
        progress = (trainer.current_epoch + 1) / self.decay_warmup_epochs
        progress = min(1.0, max(0.0, progress))
        return 0.5 + (self.decay - 0.5) * progress

    def on_train_begin(self, trainer: Any) -> None:
        """初始化 EMA 影子参数（或为已恢复的影子补齐模型张量引用）。

        只为**浮点**参数与浮点 buffer 建立影子：整型 buffer（如 BatchNorm 的
        ``num_batches_tracked``）是计数器，做滑动平均没有意义。

        同时缓存「影子张量 -> 模型张量」的引用对：``on_batch_end`` 每步都要更新，
        若每步都重新调用 ``model.state_dict()`` 会不断构造字典。缓存后每步只做
        原地算术。

        **resume 语义**：从 checkpoint 恢复时 ``load_state_dict`` 已经写入影子权重，
        此时必须保留它而不是用当前模型权重重新初始化 —— v1.0.0 会无条件重建，
        等于每次续训都把 EMA 状态清零（CodePlan 7.5 步骤 3-3 的 resume 一致性要求）。
        """
        model_state = trainer.model.state_dict()

        if self.shadow:
            # 续训路径：保留已恢复的影子，只重建引用对
            self._keys = [key for key in self._keys if key in self.shadow and key in model_state]
            self._pairs = [(self.shadow[key], model_state[key]) for key in self._keys]
            return

        self._keys = []
        self._pairs = []
        for key, value in model_state.items():
            if not value.is_floating_point():
                continue
            shadow = value.detach().clone().float()
            self._keys.append(key)
            self.shadow[key] = shadow
            self._pairs.append((shadow, value))

    def on_batch_end(
        self, trainer: Any, batch_idx: int, logs: Dict[str, float], mode: str, batch=None
    ) -> None:
        """每个训练 batch 后更新 EMA 影子权重。

        使用**原地**运算（``mul_().add_()``）。v1.0.0 写成
        ``shadow[k] = decay * shadow[k] + (1 - decay) * v.float()``，
        每步都为每个参数新建一个张量 —— 对一个 25M 参数的模型相当于每步
        多分配 100MB，实测 EMA 开销占训练总耗时 **17%**（目标 < 3%）。
        """
        if mode != "train":
            return
        with torch.no_grad():
            decay = self._effective_decay(trainer)
            weight = 1.0 - decay
            for shadow, value in self._pairs:
                shadow.mul_(decay).add_(value.detach(), alpha=weight)

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
            "keys": list(self._keys),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复 EMA 影子权重（``_pairs`` 在 ``on_train_begin`` 里重建）。"""
        self.decay = state.get("decay", self.decay)
        self.decay_warmup_epochs = state.get("decay_warmup_epochs", self.decay_warmup_epochs)
        self.shadow = state.get("shadow", self.shadow)
        keys = state.get("keys")
        # 旧格式（v1.0.0）没有 keys：从 shadow 反推，保持向后兼容
        self._keys = list(keys) if keys is not None else list(self.shadow)
        self._pairs = []


__all__ = ["EMA"]
