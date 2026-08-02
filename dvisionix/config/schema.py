# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 配置 schema 校验（轻量、无重依赖）。
"""配置 schema 校验（轻量、无重依赖）。

- 校验必填/类型/取值，错误抛 ValueError；
- 对未知配置键返回告警（避免"配了没生效"静默失败）；
- 对便捷别名（learning_rate/weight_decay）给出迁移提示。

用法：
    warnings = validate_schema(cfg.to_dict(), task_type=cfg.task_type)
    for w in warnings: logger.warning(w)
"""

from typing import Any, Dict, List, Optional

KNOWN_TOP_LEVEL = {
    "experiment_name",
    "task_type",
    "model",
    "data",
    "training",
    "checkpoint",
    "loss",
    "metrics",
    "task",
    "work_dir",
    "resume",
}

TRAINING_KEYS = {
    "num_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "device",
    "num_workers",
    "seed",
    "strategy",
    "devices",
    "amp",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "log_interval",
    "early_stopping",
    "resume_from",
    "find_unused_parameters",
}

CHECKPOINT_KEYS = {"save_dir", "monitor", "mode", "save_best_only", "save_last"}

EARLY_STOPPING_KEYS = {
    "enabled",
    "monitor",
    "mode",
    "patience",
    "min_delta",
    "restore_best_weights",
}

TASK_TYPES = ("classification", "detection", "segmentation")

_INT_KEYS = ("num_epochs", "batch_size", "num_workers", "log_interval", "accumulate_grad_batches")
_NUM_KEYS = ("learning_rate", "weight_decay")
# 允许为 null 的数值键（null 表示关闭该能力）
_OPTIONAL_NUM_KEYS = ("gradient_clip_val",)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_schema(config: Dict[str, Any], task_type: Optional[str] = None) -> List[str]:
    """校验配置字典，返回未知键/别名等告警列表；类型/取值错误抛 ValueError。"""
    warnings: List[str] = []

    if not isinstance(config, dict):
        raise ValueError("配置必须是 dict")

    for key in config:
        if key not in KNOWN_TOP_LEVEL:
            warnings.append(f"未知顶层配置键: '{key}'（将被忽略）")

    if task_type is not None and task_type not in TASK_TYPES:
        raise ValueError(f"未知 task_type: {task_type!r}（可选: {TASK_TYPES}）")

    # ---------------- training ----------------
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training 必须是 dict")

    for key in training:
        if key not in TRAINING_KEYS:
            warnings.append(f"未知 training 配置键: '{key}'（将被忽略）")

    for key in _INT_KEYS:
        if key in training and not _is_int(training[key]):
            raise ValueError(f"training.{key} 必须是整数，当前: {training[key]!r}")
    for key in _NUM_KEYS:
        if key in training and not _is_num(training[key]):
            raise ValueError(f"training.{key} 必须是数字，当前: {training[key]!r}")
    for key in _OPTIONAL_NUM_KEYS:
        if key in training and training[key] is not None and not _is_num(training[key]):
            raise ValueError(f"training.{key} 必须是数字或 null，当前: {training[key]!r}")

    if training.get("accumulate_grad_batches", 1) < 1:
        raise ValueError("training.accumulate_grad_batches 必须 >= 1")

    devices = training.get("devices")
    if devices is not None:
        if not isinstance(devices, list) or not all(_is_int(d) for d in devices):
            raise ValueError("training.devices 必须是整数列表（如 [0, 1]）或 null")

    # 别名提示：learning_rate/weight_decay 与 optimizer.lr/weight_decay 同时存在
    optimizer = training.get("optimizer")
    if isinstance(optimizer, dict) and "learning_rate" in training and "lr" in optimizer:
        warnings.append(
            "training.learning_rate 与 training.optimizer.lr 同时存在，optimizer.lr 生效（learning_rate 为便捷别名）"
        )
    if isinstance(optimizer, dict) and "weight_decay" in training and "weight_decay" in optimizer:
        warnings.append(
            "training.weight_decay 与 training.optimizer.weight_decay 同时存在，optimizer.weight_decay 生效（weight_decay 为便捷别名）"
        )

    # ---------------- checkpoint ----------------
    checkpoint = config.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 必须是 dict")
    for key in checkpoint:
        if key not in CHECKPOINT_KEYS:
            warnings.append(f"未知 checkpoint 配置键: '{key}'（将被忽略）")
    mode = checkpoint.get("mode")
    if mode is not None and mode not in ("min", "max"):
        raise ValueError("checkpoint.mode 必须是 'min' 或 'max'")

    # ---------------- early_stopping ----------------
    es = training.get("early_stopping", {})
    if not isinstance(es, dict):
        raise ValueError("training.early_stopping 必须是 dict")
    for key in es:
        if key not in EARLY_STOPPING_KEYS:
            warnings.append(f"未知 early_stopping 配置键: '{key}'（将被忽略）")
    es_mode = es.get("mode")
    if es_mode is not None and es_mode not in ("min", "max"):
        raise ValueError("training.early_stopping.mode 必须是 'min' 或 'max'")

    # ---------------- model / data ----------------
    model = config.get("model", {})
    if isinstance(model, dict) and "num_classes" in model and not _is_int(model["num_classes"]):
        raise ValueError("model.num_classes 必须是整数")

    data = config.get("data", {})
    if isinstance(data, dict) and "image_size" in data and not _is_int(data["image_size"]):
        raise ValueError("data.image_size 必须是整数")

    return warnings


__all__ = ["validate_schema"]
