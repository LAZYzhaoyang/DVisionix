# -*- coding: utf-8 -*-
"""
ONNX 模型导出器

将训练好的 PyTorch 模型导出为 ONNX 格式，并可选验证精度一致性。
"""

import os
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn


class ONNXExporter:
    """
    ONNX 模型导出器

    支持动态 batch、动态输入尺寸，并可用 onnxruntime 验证导出精度。

    Examples:
        >>> exporter = ONNXExporter(model, input_shape=(3, 32, 32))
        >>> exporter.export("model.onnx", dynamic_batch=True)
        >>> exporter.verify("model.onnx")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.input_shape = input_shape
        self.device = torch.device(device)

    def export(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        dynamic_size: bool = False,
        opset_version: int = 17,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        simplify: bool = False,
    ) -> str:
        """导出模型为 ONNX 格式，返回导出文件路径。"""
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        dummy_input = torch.randn(1, *self.input_shape, device=self.device)

        dynamic_axes: Dict[str, Dict[int, str]] = {}
        if dynamic_batch:
            dynamic_axes[input_names[0]] = {0: "batch_size"}
            dynamic_axes[output_names[0]] = {0: "batch_size"}
        if dynamic_size:
            dynamic_axes.setdefault(input_names[0], {})
            dynamic_axes[input_names[0]][2] = "height"
            dynamic_axes[input_names[0]][3] = "width"

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            dynamo=False,
        )

        print(f"[OK] ONNX model exported: {output_path}")

        if simplify:
            self._simplify(output_path)

        return output_path

    def _simplify(self, onnx_path: str) -> None:
        """使用 onnx-simplifier 简化模型（可选）。"""
        try:
            import onnx
            from onnxsim import simplify

            model_onnx = onnx.load(onnx_path)
            model_simplified, check = simplify(model_onnx)
            if check:
                onnx.save(model_simplified, onnx_path)
                print(f"[OK] ONNX model simplified: {onnx_path}")
            else:
                print("[WARN] ONNX simplify check failed, keep original model")
        except ImportError:
            print("[WARN] onnxsim not installed, skip simplify. pip install onnxsim")

    def verify(
        self,
        onnx_path: str,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        num_samples: int = 3,
    ) -> bool:
        """验证 ONNX 输出与 PyTorch 输出的一致性。"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("[WARN] onnxruntime not installed, skip verify. pip install onnxruntime")
            return False

        import numpy as np

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        all_passed = True
        for i in range(num_samples):
            dummy_input = torch.randn(1, *self.input_shape, device=self.device)
            with torch.no_grad():
                torch_output = self.model(dummy_input).cpu().numpy()
            onnx_output = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]

            is_close = np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
            max_diff = float(np.abs(torch_output - onnx_output).max())
            status = "[OK]" if is_close else "[FAIL]"
            print(f"  {status} Sample {i + 1}: max_diff={max_diff:.2e}")
            if not is_close:
                all_passed = False

        if all_passed:
            print("[OK] ONNX accuracy verification passed")
        else:
            print("[FAIL] ONNX accuracy verification failed")
        return all_passed
