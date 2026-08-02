# -*- coding: utf-8 -*-
"""Checkpoint 权重加载工具。"""

from typing import Any, Dict

import torch
import torch.nn as nn


def load_backbone(
    model: nn.Module,
    path: str,
    prefix: str = "backbone.",
    device: str = "cpu",
) -> Dict[str, Any]:
    """从预训练 checkpoint 加载骨干权重到 ``model.backbone``。

    checkpoint 兼容两种格式：
    - Trainer 完整 checkpoint（含 "model_state_dict"）；
    - 纯 state_dict（如 EMA 导出 ``ema_last.pt``）。

    自动过滤并去除 ``backbone.`` 前缀；按 key 匹配，缺失/多余键打印警告。
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            mapped[k[len(prefix) :]] = v
    if not mapped:
        mapped = dict(state)  # 兜底：视为 backbone 自身的 state_dict
    missing, unexpected = model.backbone.load_state_dict(mapped, strict=False)
    if missing:
        print(f"[load_backbone] 缺失键 {len(missing)}（示例：{missing[:3]}）")
    if unexpected:
        print(f"[load_backbone] 多余键 {len(unexpected)}（示例：{unexpected[:3]}）")
    return {"missing": missing, "unexpected": unexpected}


__all__ = ["load_backbone"]
