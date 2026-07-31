# -*- coding: utf-8 -*-
"""训练级日志器（TrainingLogger）。

统一训练过程的可观测出口：结构化日志（console + file）+ TensorBoard + JSONL 事件流。
Trainer 与所有回调都通过 ``trainer.logger`` 输出，避免散落的 print。
"""

import json
import os
from typing import Any, Dict, Optional

from .logger import get_logger, format_metrics
from .tensorboard import TensorBoardWriter


class TrainingLogger:
    """训练过程日志器。

    Args:
        name: 日志器名称。
        log_dir: 日志目录（None 时仅 console）。
        tb_dir: TensorBoard 目录（None 时不启用 TB）。
        level: 日志级别。
        console: 是否输出到控制台。
    """

    def __init__(
        self,
        name: str = "dvisionix",
        log_dir: Optional[str] = None,
        tb_dir: Optional[str] = None,
        level: str = "info",
        console: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.logger = get_logger(name, level=level, log_dir=log_dir, console=console)
        self.tb = TensorBoardWriter(tb_dir) if tb_dir else TensorBoardWriter(None)
        self.jsonl_path = os.path.join(log_dir, "events.jsonl") if log_dir else None
        if self.jsonl_path is not None:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 基础委托
    # ------------------------------------------------------------------
    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    def log_metrics(self, step: int, mode: str, metrics: Dict[str, float], precision: int = 4) -> None:
        """记录一组指标：console 摘要 + JSONL 事件 + TensorBoard 标量。"""
        self.logger.info(f"[{mode}][step {step}] {format_metrics(metrics, precision)}")

        if self.jsonl_path is not None:
            self._write_jsonl({
                "event": "metrics",
                "step": step,
                "mode": mode,
                "metrics": {k: round(float(v), 8) for k, v in metrics.items()},
            })

        if self.tb.enabled:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{mode}/{k}", float(v), step)

    def log_hparams(self, params: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """记录超参数（TensorBoard）。"""
        self.tb.add_hparams(params, metrics)

    def log_model_graph(self, model: Any, dummy_input: Any) -> None:
        """记录模型计算图（TensorBoard）。"""
        self.tb.add_graph(model, dummy_input)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """记录单张图像（TensorBoard）。"""
        self.tb.add_image(tag, image, step)

    # ------------------------------------------------------------------
    # 事件流
    # ------------------------------------------------------------------
    def log_event(self, event: str, **fields: Any) -> None:
        """写一条通用 JSONL 事件。"""
        if self.jsonl_path is not None:
            self._write_jsonl({"event": event, **fields})

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover
            pass

    def close(self) -> None:
        """关闭 TensorBoard writer 与日志文件句柄。"""
        self.tb.close()
        for handler in list(self.logger.handlers):
            try:
                handler.flush()
                handler.close()
            except Exception:  # pragma: no cover
                pass
            self.logger.removeHandler(handler)


__all__ = ["TrainingLogger"]