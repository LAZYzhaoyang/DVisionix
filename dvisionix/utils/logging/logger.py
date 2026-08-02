# -*- coding: utf-8 -*-
"""
结构化日志系统（唯一权威）

提供统一的日志器，支持控制台 + 文件双输出、日志分级、
以及按训练阶段/指标记录的便捷方法，便于排查问题。

- ``get_logger``：创建/获取配置好的 ``logging.Logger``（console + file）。
- ``format_metrics`` / ``log_metrics``：指标格式化与按阶段记录。
- JSONL 事件流由 ``TrainingLogger``（training.py）负责，不在此处。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_DEFAULT_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "dvisionix",
    level: str = "info",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """
    创建/获取一个配置好的日志器。

    Args:
        name: 日志器名称。
        level: 日志级别（debug/info/warning/error/critical）。
        log_dir: 日志目录；提供后会自动生成带时间戳的日志文件。
        log_file: 显式日志文件路径（优先级高于 log_dir 自动命名）。
        console: 是否输出到控制台。
        fmt: 日志格式。

    Returns:
        已配置的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(_LEVELS.get(level.lower(), logging.INFO))
    # 避免重复添加 handler（多次调用时）
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    target_file = log_file
    if target_file is None and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    if target_file is not None:
        parent = os.path.dirname(os.path.abspath(target_file))
        os.makedirs(parent, exist_ok=True)
        fh = logging.FileHandler(target_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.log_file = target_file  # type: ignore[attr-defined]

    return logger


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """将指标字典格式化为可读字符串，如 'loss: 0.1234 | acc: 95.20'。"""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.{precision}f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    step: Optional[int] = None,
    stage: str = "train",
    precision: int = 4,
) -> None:
    """
    按阶段记录一组指标。

    Args:
        logger: 日志器。
        metrics: 指标字典。
        step: 步数/epoch（可选）。
        stage: 阶段标签，如 'train' / 'val' / 'test'。
        precision: 浮点精度。
    """
    prefix = f"[{stage}]"
    if step is not None:
        prefix += f"[step {step}]"
    logger.info(f"{prefix} {format_metrics(metrics, precision)}")


__all__ = ["get_logger", "format_metrics", "log_metrics"]
