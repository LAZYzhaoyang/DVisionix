# 模型导出（ONNX）

`dvisionix.export.ONNXExporter` 支持将模型导出为 ONNX，并用 onnxruntime 验证精度。
支持单输入 / 多输入 / dict 输出 / 自定义 dummy 输入，以及 trace 与 dynamo 两种后端。

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

## 支持不同模型的导出

### 1. 单输入分类模型（默认路径）

```python
exporter = ONNXExporter(model, input_shape=(3, 224, 224))
exporter.export("cls.onnx")
```

### 2. 检测模型（GridDetectionModel）

```python
from dvisionix.models import GridDetectionModel
exporter = ONNXExporter(GridDetectionModel(num_classes=3), input_shape=(3, 64, 64))
exporter.export("det.onnx")
```

> 注意：检测模型导出的是 **forward 的原始预测张量**（不含 decode/NMS），
> 后处理（阈值/NMS）需要调用方在推理端自行实现。

### 3. 自定义多输入模型

```python
exporter = ONNXExporter(model, input_shapes=[(3, 224, 224), (8,)])  # 或传 dummy_inputs
exporter.export("multi.onnx", input_names=["image", "feature"])
```

### 4. dict 输出模型

模型 forward 返回 `{"logits": ..., "feat": ...}` 时，导出器自动按键生成输出名
（`logits` / `feat`），`verify` 会逐输出对比。

### 5. 自定义任意输入

```python
exporter = ONNXExporter(model, dummy_inputs=(torch.randn(2, 8), torch.randn(2, 4)))
exporter.export("custom.onnx", dynamic_batch=False)
```

## 参数说明
- `dynamic_batch`: 支持动态 batch 维度（推理时可变 batch）。
- `dynamic_size`: 支持动态 H/W（分割等任务常用）。
- `opset_version`: ONNX opset，默认 17。
- `simplify`: 是否用 onnxsim 简化计算图（需 `pip install onnxsim`）。
- `backend`: 导出后端，`'trace'`（默认，TorchScript 路径，零额外依赖）或 `'dynamo'`
  （torch.export 新路径，需 `pip install onnxscript`；对含控制流/动态 shape 的自定义模型更稳）。
- `normalize` / `metadata`: 写入 ONNX `metadata_props` 的归一化参数或附加元数据，
  推理端可直接读取做预处理。

## 注意事项
- trace 后端在 PyTorch ≥2.9 会提示 TorchScript 导出已废弃（仍可用）；复杂自定义模型建议 `backend='dynamo'`。
- `verify` 会随机采样对比 PyTorch 与 ONNX 输出（多输入/多输出逐一对比），最大误差通常在 1e-6 量级以内。
- TensorRT 导出规划中：可先导出 ONNX，再用 `trtexec` 或 torch2trt 转换。