# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: TensorRT 导出器占位（P5 扩展点）
"""TensorRT 导出器占位实现。

P5 只交付 ONNX 导出；本模块提供**可注册、可构建、可被探测**的 TensorRT
扩展点：``EXPORTERS`` 中能看到 ``"tensorrt"``，但真正调用 ``export`` 时会抛出
明确的 :class:`NotImplementedError`，避免调用方误以为已经支持。
"""

from __future__ import annotations

from typing import Any

from .base import EXPORTERS, BaseExporter, ExportResult


@EXPORTERS.register(name="tensorrt")
class TensorRTExporter(BaseExporter):
    """TensorRT 导出器扩展点（尚未实现）。

    Args:
        model: 待导出的模型对象。
        **kwargs: 预留的导出选项，原样保存在 :attr:`options` 中。
    """

    format = "tensorrt"

    def export(self, output_path: str, **kwargs: Any) -> ExportResult:
        """预留接口；当前恒抛 :class:`NotImplementedError`。

        Args:
            output_path: 预留的导出文件路径。
            **kwargs: 预留的导出选项。

        Raises:
            NotImplementedError: 始终抛出，提示 TensorRT 导出尚未实现。
        """
        raise NotImplementedError("TensorRT 导出尚未实现：本类仅作为扩展点占位（P5 只交付 ONNX）。")


__all__ = ["TensorRTExporter"]
