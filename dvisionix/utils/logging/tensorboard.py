# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: TensorBoard 轻量封装（可选依赖，缺失时自动降级为 no-op）。
"""TensorBoard 轻量封装（可选依赖，缺失时自动降级为 no-op）。"""

import os
from typing import Any, Dict, Optional

TENSORBOARD_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class TensorBoardWriter:
    """对 torch.utils.tensorboard.SummaryWriter 的薄封装。

    Args:
        log_dir: 日志目录（None 时不启用）。
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.enabled = bool(log_dir) and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录单个标量曲线。"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_dict: Dict[str, float], step: int) -> None:
        """记录多个标量到同一图。"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_dict, step)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        """记录参数/激活的直方图。"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def add_image(self, tag: str, image: Any, step: int) -> None:
        """记录图像。"""
        if self.enabled:
            self.writer.add_image(tag, image, step)

    def add_graph(self, model: Any, dummy_input: Any) -> None:
        """记录模型计算图。"""
        if self.enabled:
            try:
                self.writer.add_graph(model, dummy_input)
            except Exception:  # pragma: no cover - 图导出失败不影响训练
                pass

    def add_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """记录超参数与指标对比表。"""
        if self.enabled:
            try:
                self.writer.add_hparams(hparam_dict, metric_dict)
            except Exception:  # pragma: no cover
                pass

    def flush(self) -> None:
        """刷新写入磁盘。"""
        if self.enabled:
            self.writer.flush()

    def close(self) -> None:
        """关闭 writer。"""
        if self.enabled:
            self.writer.close()


__all__ = ["TensorBoardWriter", "TENSORBOARD_AVAILABLE"]
