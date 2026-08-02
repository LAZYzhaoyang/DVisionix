# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ONNX 模型导出器
"""
ONNX 模型导出器

支持：
- 单输入 / 多输入（input_shapes 列表或自定义 dummy_inputs）
- 输出自适应：Tensor / tuple / list / dict（dict 按键命名输出）
- 多输出 verify（onnxruntime 逐输出对比）
- trace（默认，零额外依赖）与 dynamo（可选，需 onnxscript）两种导出后端
- 任务感知：写入 transforms 归一化元数据（mean/std/scale）到 ONNX metadata_props
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ..utils import get_logger

_logger = get_logger("dvisionix.export")

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor], Sequence[torch.Tensor]]


def _flatten_outputs(outputs: TensorOrDict) -> Tuple[List[torch.Tensor], List[str]]:
    """把模型输出归一化为 (有序 Tensor 列表, 名称列表)。"""
    if isinstance(outputs, torch.Tensor):
        return [outputs], ["output"]
    if isinstance(outputs, dict):
        names = list(outputs.keys())
        tensors = [outputs[k] for k in names]
        return tensors, [str(n) for n in names]
    if isinstance(outputs, (list, tuple)):
        tensors = list(outputs)
        return tensors, [f"output_{i}" for i in range(len(tensors))]
    raise TypeError(f"不支持的模型输出类型: {type(outputs)}（期望 Tensor / dict / list / tuple）")


class ONNXExporter:
    """ONNX 模型导出器。

    Args:
        model: 待导出的 PyTorch 模型（forward 需可被 trace）。
        input_shape: 单输入模型输入形状（兼容旧 API），如 ``(3, 32, 32)``。
        input_shapes: 多输入模型输入形状列表，如 ``[(3, 32, 32), (8,)]``。
        dummy_inputs: 自定义 dummy 输入（Tensor 或 Tensor 元组/列表），非 image 输入用这个。
        device: 导出设备（'cpu' / 'cuda'）。
        task_type: 任务类型（'classification' / 'detection' / 'segmentation'，可选）。
        normalize: 归一化元数据，如 ``{"mean": [...], "std": [...], "scale": 1/255}``。

    Examples:
        >>> exporter = ONNXExporter(model, input_shape=(3, 32, 32))
        >>> exporter.export("model.onnx", dynamic_batch=True)
        >>> exporter.verify("model.onnx")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_shapes: Optional[List[Tuple[int, ...]]] = None,
        dummy_inputs: Any = None,
        device: str = "cpu",
        task_type: Optional[str] = None,
        normalize: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.task_type = task_type
        self.normalize = normalize

        if dummy_inputs is not None:
            self.dummy_inputs = (
                dummy_inputs if isinstance(dummy_inputs, (list, tuple)) else (dummy_inputs,)
            )
        elif input_shapes is not None:
            self.dummy_inputs = tuple(torch.randn(1, *s, device=self.device) for s in input_shapes)
        elif input_shape is not None:
            self.dummy_inputs = (torch.randn(1, *input_shape, device=self.device),)
        else:
            raise ValueError("必须提供 input_shape / input_shapes / dummy_inputs 之一")

        self.single_input = len(self.dummy_inputs) == 1
        # 兼容旧 API：self.input_shape
        self.input_shape = tuple(self.dummy_inputs[0].shape[1:]) if self.single_input else None
        self.input_shapes = [tuple(t.shape[1:]) for t in self.dummy_inputs]

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def export(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        dynamic_size: bool = False,
        opset_version: int = 17,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        simplify: bool = False,
        backend: str = "trace",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """导出模型为 ONNX 格式，返回导出文件路径。

        Args:
            backend: 'trace'（默认）或 'dynamo'（需安装 onnxscript）。
            metadata: 附加元数据，与 normalize 合并写入 ONNX metadata_props。
        """
        if backend not in ("trace", "dynamo"):
            raise ValueError(f"backend 必须是 'trace' 或 'dynamo'，当前: {backend!r}")

        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        # 探测输出结构（dict / 多输出）
        with torch.no_grad():
            outputs = self.model(*self.dummy_inputs)
        out_tensors, default_output_names = _flatten_outputs(outputs)

        n_inputs = len(self.dummy_inputs)
        input_names = input_names or (
            [f"input_{i}" for i in range(n_inputs)] if n_inputs > 1 else ["input"]
        )
        output_names = output_names or default_output_names
        if len(output_names) != len(out_tensors):
            raise ValueError(
                f"output_names 数量 ({len(output_names)}) 与模型输出数 ({len(out_tensors)}) 不一致"
            )

        dynamic_axes: Dict[str, Dict[int, str]] = {}
        if dynamic_batch:
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size"}
            for name in output_names:
                dynamic_axes.setdefault(name, {})[0] = "batch_size"
        if dynamic_size:
            for i, name in enumerate(input_names):
                shape = self.input_shapes[i]
                if len(shape) >= 3:  # (C, H, W, ...)
                    axes = dynamic_axes.setdefault(name, {})
                    axes[2] = "height"
                    axes[3] = "width"

        args = self.dummy_inputs[0] if self.single_input else self.dummy_inputs

        if backend == "dynamo":
            try:
                import onnxscript  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "backend='dynamo' 需要安装 onnxscript：pip install onnxscript"
                ) from exc
            torch.onnx.export(
                self.model,
                args,
                output_path,
                opset_version=opset_version,
                dynamo=True,
            )
        else:
            torch.onnx.export(
                self.model,
                args,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if dynamic_axes else None,
                dynamo=False,
            )

        self._write_metadata(output_path, metadata)
        _logger.info(f"[OK] ONNX model exported: {output_path}")

        if simplify:
            self._simplify(output_path)

        return output_path

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    def _write_metadata(self, onnx_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        props: Dict[str, Any] = {}
        if self.normalize:
            props.update({f"normalize_{k}": json.dumps(v) for k, v in self.normalize.items()})
        if metadata:
            props.update(
                {
                    str(k): json.dumps(v) if not isinstance(v, str) else v
                    for k, v in metadata.items()
                }
            )
        if not props:
            return
        try:
            import onnx

            model = onnx.load(onnx_path)
            for key, value in props.items():
                prop = model.metadata_props.add()
                prop.key = key
                prop.value = str(value)
            onnx.save(model, onnx_path)
        except ImportError:  # pragma: no cover
            _logger.warning("onnx 未安装，跳过 metadata 写入")

    # ------------------------------------------------------------------
    # 简化（可选）
    # ------------------------------------------------------------------
    def _simplify(self, onnx_path: str) -> None:
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(onnx_path)
            model_simplified, check = simplify(model_onnx)
            if check:
                onnx.save(model_simplified, onnx_path)
                _logger.info(f"[OK] ONNX model simplified: {onnx_path}")
            else:
                _logger.warning("[WARN] ONNX simplify check failed, keep original model")
        except ImportError:  # pragma: no cover
            _logger.warning("[WARN] onnxsim not installed, skip simplify. pip install onnxsim")

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    def verify(
        self,
        onnx_path: str,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        num_samples: int = 3,
    ) -> bool:
        """验证 ONNX 输出与 PyTorch 输出的一致性（多输入/多输出）。"""
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            _logger.warning(
                "[WARN] onnxruntime not installed, skip verify. pip install onnxruntime"
            )
            return False

        import numpy as np

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_names = [inp.name for inp in session.get_inputs()]
        onnx_output_names = [out.name for out in session.get_outputs()]

        all_passed = True
        for sample_idx in range(num_samples):
            dummy = tuple(t.clone() for t in self.dummy_inputs)
            with torch.no_grad():
                torch_outs = _flatten_outputs(self.model(*dummy))[0]

            feed = {name: t.numpy() for name, t in zip(onnx_input_names, dummy)}
            onnx_outs = session.run(None, feed)

            for out_idx, (torch_out, onnx_out) in enumerate(zip(torch_outs, onnx_outs)):
                onnx_name = (
                    onnx_output_names[out_idx]
                    if out_idx < len(onnx_output_names)
                    else f"#{out_idx}"
                )
                is_close = np.allclose(torch_out.numpy(), onnx_out, rtol=rtol, atol=atol)
                max_diff = float(np.abs(torch_out.numpy() - onnx_out).max())
                status = "[OK]" if is_close else "[FAIL]"
                _logger.info(
                    f"  {status} Sample {sample_idx + 1} output '{onnx_name}': max_diff={max_diff:.2e}"
                )
                if not is_close:
                    all_passed = False

        if all_passed:
            _logger.info("[OK] ONNX accuracy verification passed")
        else:
            _logger.error("[FAIL] ONNX accuracy verification failed")
        return all_passed


__all__ = ["ONNXExporter"]
