# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 模型导出模块（P5 统一导出接口）
"""模型导出模块。

支持把训练好的 PyTorch 模型导出为 ONNX 格式并验证精度一致性；P5 起提供统一的
导出接口：:class:`BaseExporter` + :data:`EXPORTERS` 注册表 +
:class:`ExportResult` / :class:`VerifyReport`，并预留 :class:`TensorRTExporter`
扩展点。

兼容性：``from dvisionix.export import ONNXExporter`` 与旧构造签名保持不变；
``ONNXExporter.export`` 现在返回 :class:`ExportResult`（其 ``__str__`` /
``__fspath__`` 均返回路径），``verify`` 返回 :class:`VerifyReport`（``bool()``
等价于 ``passed``）。
"""

from .base import EXPORTERS, BaseExporter, ExportResult, VerifyReport, build_exporter
from .onnx_exporter import ONNXExporter
from .tensorrt import TensorRTExporter

__all__ = [
    "ONNXExporter",
    "BaseExporter",
    "ExportResult",
    "VerifyReport",
    "EXPORTERS",
    "build_exporter",
    "TensorRTExporter",
]
