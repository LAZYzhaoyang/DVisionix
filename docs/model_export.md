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
---

## v1.1 变更要点

> 完整清单见 [v1.1 变更与迁移指南](v1.1_changes.md)。

**任意嵌套输出都支持。** v1.0.0 的 `_flatten_outputs` 只处理一层：
检测模型常见的 `{"boxes": [t0, t1], ...}` 或
`(boxes_list, scores_list, labels_list)` 会把 **list 本身**当成 Tensor 传给
`.numpy()` 而直接崩溃。现在递归展平，并把「输出名 → 访问路径」映射写进
ONNX `metadata_props.output_path_map`：

```python
import json, onnx
model = onnx.load("model.onnx")
props = {p.key: p.value for p in model.metadata_props}
print(json.loads(props["output_path_map"]))
# {'cls_0': 'cls[0]', 'cls_1': 'cls[1]', 'box_main': 'box.main'}
```

**导出器不再修改调用者模型。** v1.0.0 在 `__init__` 里直接
`model.to(device).eval()`，导出一次就会永久改变原模型的设备与 train/eval 状态。
现在只在 `export()` / `verify()` 期间临时切换，并用 `try/finally` 恢复
（**失败路径同样恢复**）。

**verify 更严格。** 统一 `detach().cpu().numpy()`（CUDA 或 requires_grad 的张量
原本会抛错）；输出数量或形状不一致时**显式报错**，而不是被 `zip` 静默截断。

**dynamo 后端的参数约束显式化。** dynamo 导出用 `dynamic_shapes` 描述动态维度，
不支持 `dynamic_axes`：

```python
# 会抛 ValueError 并说明原因，而不是静默忽略 dynamic_axes
exporter.export("m.onnx", backend="dynamo", dynamic_batch=True)

# 正确用法
exporter.export("m.onnx", backend="dynamo", dynamic_batch=False, dynamic_size=False)
```

另需安装 `onnxscript`（`pip install onnxscript`）。

**性能**：匈牙利匹配从 13.6 秒降到 18.6 毫秒（300 查询 × 30 GT），
详见 [v1.1 变更与迁移指南](v1.1_changes.md) 第 4 节。
