# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Checkpoint 权重加载工具。
"""Checkpoint 权重加载工具。"""

from typing import Any, Dict

import torch
import torch.nn as nn


def load_backbone(
    model: nn.Module,
    path: str,
    prefix: str = "backbone.",
    device: str = "cpu",
    trusted: bool = False,
) -> Dict[str, Any]:
    """从预训练 checkpoint 加载骨干权重到 ``model.backbone``。

    checkpoint 兼容两种格式：
    - Trainer 完整 checkpoint（含 "model_state_dict"）；
    - 纯 state_dict（如 EMA 导出 ``ema_last.pt``）。

    自动过滤并去除 ``backbone.`` 前缀；按 key 匹配，缺失/多余键打印警告。

    Args:
        trusted: **安全开关**。默认 ``False``，即使用 ``torch.load(weights_only=True)``，
            只允许反序列化张量与基本容器 —— 反序列化 pickle 等于执行任意代码，
            而预训练权重通常来自第三方，因此默认按不可信处理。
            仅当你确认文件来源可信、且它确实携带非张量状态（如优化器/RNG 状态）时，
            才显式传 ``trusted=True``。
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=not trusted)
    except Exception as exc:
        if trusted:
            raise
        raise ValueError(
            f"以安全模式（weights_only=True）加载 {path} 失败：{type(exc).__name__}: {exc}\n"
            f"该文件可能携带非张量状态（完整 Trainer checkpoint 含优化器/RNG 状态）。"
            f"如果确认文件来源可信，请显式传 trusted=True 重新加载；"
            f"若只需要骨干权重，建议先用本项目的 Trainer 重新导出纯 state_dict。"
        ) from exc
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
