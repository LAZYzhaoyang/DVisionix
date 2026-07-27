# 模型导出（ONNX）

`dvisionix.export.ONNXExporter` 支持将模型导出为 ONNX，并用 onnxruntime 验证精度。

## 导出与验证

```python
import torch
from dvisionix.models import SimpleCNN
from dvisionix.export import ONNXExporter

model = SimpleCNN(num_classes=10)
exporter = ONNXExporter(model, input_shape=(3, 32, 32), device="cpu")

exporter.export("./exports/simple_cnn.onnx", dynamic_batch=True, opset_version=17)
exporter.verify("./exports/simple_cnn.onnx", num_samples=3)
```

一键运行示例：

```bash
conda run -n dvisionix python demos/export_onnx_demo.py
```

## 参数说明
- `dynamic_batch`: 支持动态 batch 维度（推理时可变 batch）。
- `dynamic_size`: 支持动态 H/W（分割等任务常用）。
- `opset_version`: ONNX opset，默认 17。
- `simplify`: 是否用 onnxsim 简化计算图（需 `pip install onnxsim`）。

## 注意事项
- 本项目导出使用稳定的 TorchScript 路径（`dynamo=False`），无需额外安装 `onnxscript`。
- `verify` 会随机采样对比 PyTorch 与 ONNX 输出，最大误差通常在 1e-6 量级以内。
- TensorRT 导出规划中：可先导出 ONNX，再用 `trtexec` 或 torch2trt 转换。
