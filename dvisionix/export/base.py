# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 导出器公共契约（注册表、导出结果、校验报告与抽象基类）
"""导出器公共契约：注册表、导出结果、校验报告与抽象基类。

本模块是 P5 导出体系的**唯一接口层**：

- :data:`EXPORTERS`：全局导出器注册表，新格式用
  ``@EXPORTERS.register(name="xxx")`` 注册即可被 :func:`build_exporter` 构建；
- :class:`ExportResult`：导出产物的统一描述（路径、输入输出、opset、
  动态轴、预处理规格与数值一致性等），支持 JSON sidecar 落盘/加载；
- :class:`VerifyReport`：导出产物数值校验的统一报告，``bool(report)``
  等价于 :attr:`VerifyReport.passed`，兼容既有 ``assert exporter.verify(path)``；
- :class:`BaseExporter`：所有导出器的抽象基类。

典型用法::

    from dvisionix.export import build_exporter

    exporter = build_exporter("onnx", model, input_shape=(3, 32, 32))
    result = exporter.export("model.onnx", dynamic_batch=True)
    result.save("model.onnx.json")
    assert exporter.verify(result)
"""

from __future__ import annotations

import dataclasses
import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from ..errors import ConfigError
from ..registry import Registry

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查，避免导出层在导入期拉起推理依赖
    from ..inference.preprocess import PreprocessSpec

#: 全局导出器注册表：注册名（``"onnx"`` / ``"tensorrt"`` / ...）-> 导出器类
EXPORTERS: Registry = Registry("exporters")


@dataclasses.dataclass
class ExportResult:
    """一次导出的结果描述（可作为 JSON sidecar 落盘/加载）。

    Attributes:
        path: 导出文件路径；``str(result)`` / ``os.fspath(result)`` / ``open(result)``
            均直接使用该路径，因此既有把返回值当路径使用的代码保持可用。
        format: 导出格式名（``"onnx"`` / ``"tensorrt"`` / ...）。
        input_names: 导出模型的实际输入名（顺序与 dummy 输入一致）。
        input_shapes: 每个 dummy 输入的**完整形状**（含 batch 维）。
        output_names: 导出模型的实际输出名。
        opset: ONNX opset 版本；非 ONNX 格式为 None。
        dynamic_axes: 实际生效的动态轴映射（``{张量名: {轴: 标签}}``）；无则 None。
        preprocess: 推理预处理规格（:class:`~dvisionix.inference.preprocess.PreprocessSpec`）。
        parity_max_abs_diff: 与 PyTorch 输出比较的全局最大绝对误差，``verify`` 后回填。
        extra: 格式相关的附加信息（如后端、是否简化、输出路径映射）。
    """

    path: str
    format: str
    input_names: List[str]
    input_shapes: List[List[int]]
    output_names: List[str]
    opset: Optional[int] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    preprocess: Optional["PreprocessSpec"] = None
    parity_max_abs_diff: Optional[float] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的字典（``preprocess`` 转为其 ``to_dict()`` 产物）。

        Returns:
            仅含基础类型（str / int / float / bool / None / list / dict）的字典，
            可直接交给 ``json.dump``。
        """
        return {
            "path": self.path,
            "format": self.format,
            "input_names": list(self.input_names),
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_names": list(self.output_names),
            "opset": self.opset,
            "dynamic_axes": (
                {str(name): dict(axes) for name, axes in self.dynamic_axes.items()}
                if self.dynamic_axes is not None
                else None
            ),
            "preprocess": (self.preprocess.to_dict() if self.preprocess is not None else None),
            "parity_max_abs_diff": self.parity_max_abs_diff,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportResult":
        """从 :meth:`to_dict` 产物（或等价字典）还原结果。

        兼容 JSON 往返：``dynamic_axes`` 的轴键在 JSON 中会变成字符串，这里统一
        转回 ``int``；``preprocess`` 字典会还原为
        :class:`~dvisionix.inference.preprocess.PreprocessSpec`。

        Args:
            data: 至少包含 ``path`` / ``format`` / ``input_names`` /
                ``input_shapes`` / ``output_names`` 的字典。

        Returns:
            还原后的导出结果。

        Raises:
            TypeError: ``data`` 不是字典。
            KeyError: 缺少必填字段。
        """
        if not isinstance(data, dict):
            raise TypeError(f"ExportResult.from_dict 需要 dict，实际为 {type(data).__name__}")

        dynamic_axes = data.get("dynamic_axes")
        if dynamic_axes is not None:
            dynamic_axes = {
                str(name): {int(axis): str(label) for axis, label in axes.items()}
                for name, axes in dynamic_axes.items()
            }

        preprocess = data.get("preprocess")
        if isinstance(preprocess, dict):
            from ..inference.preprocess import PreprocessSpec

            preprocess = PreprocessSpec.from_dict(preprocess)

        return cls(
            path=str(data["path"]),
            format=str(data["format"]),
            input_names=list(data["input_names"]),
            input_shapes=[list(shape) for shape in data["input_shapes"]],
            output_names=list(data["output_names"]),
            opset=data.get("opset"),
            dynamic_axes=dynamic_axes,
            preprocess=preprocess,
            parity_max_abs_diff=data.get("parity_max_abs_diff"),
            extra=dict(data.get("extra") or {}),
        )

    def save(self, path: str) -> None:
        """把结果写入 JSON sidecar（``indent=2, ensure_ascii=False``）。

        Args:
            path: sidecar 输出路径；父目录不存在时自动创建。
        """
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ExportResult":
        """从 :meth:`save` 写出的 JSON sidecar 加载结果。

        Args:
            path: sidecar 路径。

        Returns:
            还原后的导出结果。
        """
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def __str__(self) -> str:
        """返回导出文件路径，兼容把结果直接当字符串使用的旧代码。"""
        return self.path

    def __fspath__(self) -> str:
        """返回导出文件路径，兼容 ``open(result)`` / ``os.path.exists(result)``。"""
        return self.path


@dataclasses.dataclass
class VerifyReport:
    """导出产物与 PyTorch 输出的一致性校验报告。

    Attributes:
        passed: 是否全部输出、全部样本都通过比较。
        max_abs_diff: 全局最大绝对误差；没有可比对输出（如缺少 onnxruntime）时 None。
        outputs: 逐输出明细，每项含 ``name`` / ``shape`` / ``max_abs_diff`` / ``passed``。
    """

    passed: bool
    max_abs_diff: Optional[float] = None
    outputs: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def __bool__(self) -> bool:
        """布尔语义等价于 :attr:`passed`（``assert exporter.verify(path)`` 继续可用）。"""
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好的字典。"""
        return {
            "passed": self.passed,
            "max_abs_diff": self.max_abs_diff,
            "outputs": [dict(item) for item in self.outputs],
        }


class BaseExporter(ABC):
    """所有导出器的抽象基类。

    Args:
        model: 待导出的模型对象。
        **kwargs: 导出器通用选项，原样保存在 :attr:`options` 中。

    Attributes:
        format: 导出格式名；子类必须覆盖为非空字符串并注册到 :data:`EXPORTERS`。
        model: 待导出的模型对象。
        options: 构造时传入的附加选项。
    """

    format: ClassVar[str] = ""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.options = dict(kwargs)

    @abstractmethod
    def export(self, output_path: str, **kwargs: Any) -> ExportResult:
        """把模型导出到 ``output_path`` 并返回导出结果。

        Args:
            output_path: 导出文件路径。
            **kwargs: 各格式特有的导出选项。

        Returns:
            导出结果描述。
        """
        raise NotImplementedError

    def verify(self, result_or_path: Any, **kwargs: Any) -> VerifyReport:
        """校验导出产物与原始模型输出的一致性。

        Args:
            result_or_path: :class:`ExportResult` 或导出文件路径。
            **kwargs: 各格式特有的校验选项。

        Returns:
            校验报告。

        Raises:
            NotImplementedError: 该导出器未实现数值校验（默认行为）。
        """
        raise NotImplementedError("该导出器未实现 verify")


def build_exporter(format: str, model: Any, **kwargs: Any) -> BaseExporter:
    """按注册名构建导出器。

    Args:
        format: 已注册的导出格式名（如 ``"onnx"`` / ``"tensorrt"``）。
        model: 待导出的模型对象。
        **kwargs: 传给导出器构造函数的参数。

    Returns:
        对应的导出器实例。

    Raises:
        ConfigError: ``format`` 未在 :data:`EXPORTERS` 中注册；错误消息会列出
            全部已注册格式（不会抛裸 ``KeyError``）。
    """
    if format not in EXPORTERS:
        registered = sorted(EXPORTERS.keys())
        available = ", ".join(registered) if registered else "<无>"
        raise ConfigError(
            f"未知的导出格式：{format!r}；已注册格式：{available}",
            hint="请检查 --format 参数，或先通过 EXPORTERS.register 注册自定义导出器",
        )
    return EXPORTERS.get(format)(model, **kwargs)


__all__ = [
    "EXPORTERS",
    "ExportResult",
    "VerifyReport",
    "BaseExporter",
    "build_exporter",
]
