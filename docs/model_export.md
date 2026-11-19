# 模型导出（ONNX）

> 适用范围：v2.0.0（P5 导出体系）。本页说明如何把训练好的 checkpoint
> 导出为 ONNX、如何做数值一致性校验，以及导出体系的扩展方式。
>
> 接口定义以实现契约 [P5 导出接口契约](history/design/v2_p5_contract.md) 为准。相关文档：
> [快速开始](quick_start.md)、[推理链路](inference.md)、[目标检测](detection.md)、
> [语义分割](segmentation.md)。

## 1. 体系总览

P5 把「只能导出 ONNX」升级为**有接口、可注册、可校验、可扩展**的导出体系。
唯一接口层在 `dvisionix/export/base.py`：

```text
config + checkpoint
        │  tools/export.py（CLI）：读配置 → 建模型 → 载权重
        ▼
build_exporter(format, model, **kwargs)
        │  EXPORTERS："onnx" → ONNXExporter；"tensorrt" → TensorRTExporter（占位）
        ▼
exporter.export(path, **kwargs) ──► ExportResult
        │  .save("<output>.json") ──► JSON sidecar
        ▼
exporter.verify(result) ──► VerifyReport（bool 语义 = passed）
```

### 1.1 核心对象

| 名称 | 位置 | 职责 |
|---|---|---|
| `BaseExporter` | `dvisionix/export/base.py` | 所有导出器的抽象基类；子类实现 `export()`，可选实现 `verify()` |
| `EXPORTERS` | `dvisionix/export/base.py` | 全局导出器注册表（复用 `dvisionix.registry.Registry`），注册名 → 导出器类 |
| `build_exporter(format, model, **kwargs)` | `dvisionix/export/base.py` | 按注册名构建导出器；未知格式抛 `ConfigError`（不是裸 `KeyError`） |
| `ExportResult` | `dvisionix/export/base.py` | 一次导出的完整描述，可落盘为 JSON sidecar |
| `VerifyReport` | `dvisionix/export/base.py` | 数值一致性校验报告，`bool(report) == report.passed` |
| `ONNXExporter` | `dvisionix/export/onnx_exporter.py` | ONNX 实现，注册名 `onnx` |
| `TensorRTExporter` | `dvisionix/export/tensorrt.py` | TensorRT **扩展点占位**，注册名 `tensorrt` |

`dvisionix/export/__init__.py` 公开导出：
`ONNXExporter`、`BaseExporter`、`ExportResult`、`VerifyReport`、`EXPORTERS`、
`build_exporter`、`TensorRTExporter`。

### 1.2 已注册的导出格式

| 注册名 | 类 | 状态 | 说明 |
|---|---|---|---|
| `onnx` | `ONNXExporter` | ✅ 可用 | 支持 trace / dynamo 后端、单/多输入、任意嵌套输出、ORT 一致性校验 |
| `tensorrt` | `TensorRTExporter` | 🚧 占位 | 可注册、可构建；`export()` 恒抛 `NotImplementedError`（见第 8 节） |

查询当前注册表：

```python
from dvisionix.export import EXPORTERS
print(sorted(EXPORTERS.keys()))   # ['onnx', 'tensorrt']
```

### 1.3 最小示例

```python
import onnx
from dvisionix.models import SimpleCNN
from dvisionix.export import build_exporter

model = SimpleCNN(num_classes=10)
exporter = build_exporter(
    "onnx", model, input_shape=(3, 32, 32), device="cpu", task_type="classification"
)
result = exporter.export("model.onnx", dynamic_batch=True)
result.save("model.onnx.json")

report = exporter.verify(result)          # VerifyReport
print(report.passed, report.max_abs_diff)

print(str(result))                        # "model.onnx"（见第 4 节）
onnx.load(result)                         # os.fspath 协议生效
```

## 2. 命令行：`tools/export.py`

`tools/export.py` 是配置驱动的导出入口，流程固定为：

1. `Config.from_yaml(--config)` + `--cfg-options` 覆盖；
2. `build_model(cfg.model.to_dict())` 构建模型；
3. 加载 checkpoint（默认 `weights_only=True`，完整 Trainer checkpoint 需 `--trusted`）；
4. 从配置推导输入形状与归一化元数据（`in_channels` / `data.image_size` /
   `data.mean` / `data.std`，缺省 ImageNet + `1/255`）；
5. `build_exporter(--format, ...)` → `export()`，落盘 `<output>.json` sidecar；
6. `--verify` 时用 onnxruntime 与 PyTorch 逐输出比对，未通过以退出码 1 结束。

### 2.1 三条常用命令

以下命令与一次真实验收一一对应（1 epoch 冒烟 checkpoint，仅用于确认链路，
**不作效果结论**）。请把 `--config` 换成训练时用的配置、`--checkpoint` 换成
对应权重：

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python

# 分类：MobileNetV3 + ClsHead，5 类，224×224
$PY tools/export.py --config configs/examples/dvx_classification.yaml \
  --checkpoint /path/to/cls/best.pt --output exports/cls.onnx --verify --trusted

# 检测：YOLO，20 类，512×512，8 个输出（cls_0..3 / reg_0..3）
$PY tools/export.py --config configs/examples/dvx_detection.yaml \
  --checkpoint /path/to/det/best.pt --output exports/det.onnx --verify --trusted

# 分割：SegFormer，18 类，256×256
$PY tools/export.py --config configs/examples/dvx_segmentation.yaml \
  --checkpoint /path/to/seg/best.pt --output exports/seg.onnx --verify --trusted
```

> 想先快速试通链路，把 `--config` 换成 `configs/classification/demo_synthetic.yaml`
> （`SimpleCNN`，32×32）即可，秒级完成。
>
> 导出只读配置的 `model` / `data.image_size` / `mean` / `std` / `task_type`，
> **不构建数据集**，因此 `data.spec` 的占位路径不影响导出；`--cfg-options` 可临时
> 改输入尺寸（如 `--cfg-options data.image_size=384`，放在命令末尾）。
>
> 动态 batch：CLI 的 `--dynamic-batch` 默认**关闭**（静态 batch），需要动态 batch
> 时显式传入；这与 Python API `export(dynamic_batch=None)` 的 trace 默认值不同。

### 2.2 完整参数表

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | **必填** | YAML 配置路径；必须含 `model` 段，`data.image_size` 缺省 32 |
| `--checkpoint` | **必填** | `.pt` / `.pth`；纯 state_dict 或含 `model_state_dict` 的完整 Trainer checkpoint |
| `--output` | **必填** | 输出路径，如 `model.onnx`；父目录自动创建，sidecar 为 `<output>.json` |
| `--format` | `onnx` | 注册名；当前 `onnx` 可用、`tensorrt` 为占位 |
| `--dynamic-batch` | 关闭 | 导出动态 batch 维（axis 0 标记为 `batch_size`） |
| `--opset` | `17` | ONNX opset 版本 |
| `--verify` | 关闭 | 导出后用 onnxruntime 与 PyTorch 逐输出比对；未通过退出码 1 |
| `--trusted` | 关闭 | 允许 `torch.load(weights_only=False)` 加载完整 Trainer checkpoint（含优化器 / RNG 等非张量状态） |
| `--device` | `cpu` | 导出设备：`cpu` / `cuda`（影响 dummy 输入与临时切换的设备） |
| `--cfg-options` | 空 | 覆盖配置，如 `data.image_size=384`；`nargs="*"`，建议放在命令末尾 |

CLI **没有**暴露 `--backend` / `--simplify` / `--input-names` / `--output-names` /
`--dynamic-size`；这些选项请用 Python API（第 5 节）。CLI 固定按配置构造**单个图像
输入**，多输入模型也需走 Python API。

### 2.3 输出示例

分类（真实 checkpoint，224×224，`--verify`）的 stdout（已省略 logger 行）：

```text
一致性校验: 通过，全局 max_abs_diff=5.960e-06
  - output: shape=[1, 5] max_abs_diff=5.960e-06 passed=True
导出完成: exports/cls.onnx
元数据: exports/cls.onnx.json
格式: onnx（opset=17，动态 batch=False）
输入: ['input'] 形状: [[1, 3, 224, 224]]
输出: ['output']
预处理: {"task_type": "classification", "image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "scale": 0.00392156862745098}
```

检测（8 个输出）与分割的摘要行同样包含逐输出明细：

```text
一致性校验: 通过，全局 max_abs_diff=3.340e-06
  - cls_0: shape=[1, 20, 128, 128] max_abs_diff=… passed=True   # 共 8 条
  - reg_3: shape=[1, 4, 16, 16] max_abs_diff=… passed=True
导出完成: exports/det.onnx
输入: ['input'] 形状: [[1, 3, 512, 512]]
输出: ['cls_0', 'cls_1', 'cls_2', 'cls_3', 'reg_0', 'reg_1', 'reg_2', 'reg_3']

一致性校验: 通过，全局 max_abs_diff=4.770e-06
  - output: shape=[1, 18, 256, 256] max_abs_diff=4.770e-06 passed=True
导出完成: exports/seg.onnx
```

注意打印顺序：`--verify` 时**一致性校验在前、导出摘要在后**。上面的示例省略了 logger
行（`[OK] ONNX model exported`、逐样本 `Sample 1 output ...`）——console 日志同样写
stdout，实际终端里会与摘要交错出现。

### 2.4 sidecar 与退出码

- sidecar 文件名固定为 `f"{--output}.json"`（`exports/cls.onnx` →
  `exports/cls.onnx.json`），`indent=2, ensure_ascii=False`；
- CLI **先校验、后落盘 sidecar**：`--verify` 通过时 `parity_max_abs_diff` 已回填，
  不带 `--verify` 时为 `null`；
- `--verify` 未通过：打印「一致性校验: 未通过」，仍写出 ONNX 与 sidecar，再退出码 1；
- `--format` 未注册：`build_exporter` 抛 `ConfigError`（消息列出全部已注册格式）；
- 不带 `--trusted` 加载完整 Trainer checkpoint：抛 `CheckpointError` 并提示加
  `--trusted`；
- checkpoint 与模型结构不匹配：`CheckpointError`，提示核对 `model` 段（层名 /
  类别数 / `in_channels`）与训练时一致。

## 3. `ExportResult` 与 sidecar

### 3.1 字段表

| 字段 | 类型 / 默认值 | 含义 |
|---|---|---|
| `path` | `str` | 导出文件路径；`str(result)` / `os.fspath(result)` / `open(result)` 均使用它 |
| `format` | `str` | 导出格式名：`"onnx"` / `"tensorrt"` / 自定义 |
| `input_names` | `list[str]` | ONNX 实际输入名；单输入默认 `["input"]`，多输入从 `input_0` 起顺序命名 |
| `input_shapes` | `list[list[int]]` | 每个 dummy 输入的**完整形状**（含 batch 维），如 `[[1, 3, 224, 224]]` |
| `output_names` | `list[str]` | ONNX 实际输出名（顺序与展平后的输出一致） |
| `opset` | `int \| None` | ONNX opset；非 ONNX 格式为 `None` |
| `dynamic_axes` | `dict[str, dict[int, str]] \| None` | 实际生效的动态轴；无动态轴时 `None` |
| `preprocess` | `PreprocessSpec \| None` | 推理预处理规格（任务类型 / 尺寸 / mean / std / scale），见 5.4 |
| `parity_max_abs_diff` | `float \| None` | 与 PyTorch 比较的全局最大绝对误差；`verify()` 后回填 |
| `extra` | `dict[str, Any]` | 格式相关附加信息；ONNX 为 `backend` / `simplify` / `output_path_map` |

`preprocess` 来自 `dvisionix.inference.preprocess.PreprocessSpec`，字段为
`task_type` / `image_size` / `mean` / `std` / `scale`；它描述「输入图像如何变成网络
张量」，推理端可据此复现预处理（见 [推理链路](inference.md)）。

### 3.2 JSON sidecar 示例

```json
{
  "path": "exports/cls.onnx",
  "format": "onnx",
  "input_names": ["input"],
  "input_shapes": [[1, 3, 224, 224]],
  "output_names": ["output"],
  "opset": 17,
  "dynamic_axes": {"input": {"0": "batch_size"}, "output": {"0": "batch_size"}},
  "preprocess": {
    "task_type": "classification",
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "scale": 0.00392156862745098
  },
  "parity_max_abs_diff": 5.96e-06,
  "extra": {
    "backend": "trace",
    "simplify": false,
    "output_path_map": {"output": "output"}
  }
}
```

多输出模型（检测）的 `output_path_map` 记录「输出名 → 模型输出访问路径」，便于把
ONNX 输出与 PyTorch 返回结构对应起来：

```json
"output_path_map": {
  "cls_0": "cls[0]", "cls_1": "cls[1]", "cls_2": "cls[2]", "cls_3": "cls[3]",
  "reg_0": "reg[0]", "reg_1": "reg[1]", "reg_2": "reg[2]", "reg_3": "reg[3]"
}
```

### 3.3 序列化 API

```python
from dvisionix.export import ExportResult

data = result.to_dict()                 # preprocess 转成 to_dict() 产物
same = ExportResult.from_dict(data)     # 还原：dynamic_axes 的轴键转回 int，
                                        # preprocess 字典还原为 PreprocessSpec
result.save("model.onnx.json")          # indent=2, ensure_ascii=False，自动建父目录
loaded = ExportResult.load("model.onnx.json")
```

往返后 `loaded == result`（dataclass 值相等）。`from_dict` 对缺失的可选字段使用默认值；
`data` 不是 dict 抛 `TypeError`，缺必填字段抛 `KeyError`。

### 3.4 路径协议（兼容旧用法）

`ExportResult` 实现了字符串与路径协议：

```python
str(result); os.fspath(result)          # 均返回 result.path（str）
Path(result)                            # PosixPath(result.path)
os.path.exists(result); open(result, "rb"); onnx.load(result)   # 均可用
```

因此**仓库内没有任何调用方依赖** `export() == "path"`（契约已核实），但外部代码若写
`result == "model.onnx"`、`result.split(".")`、把结果直接塞进只接受 `str` 的第三方
接口，就需要显式 `str(result)`。

## 4. 向后兼容说明

P5 重构只改了**返回类型**与**新增一个可选构造参数**，公开导入路径与函数语义保持兼容：

| 接口 | P5 之前 | P5 之后 | 兼容性 |
|---|---|---|---|
| `ONNXExporter.export(...)` | `str`（导出路径） | `ExportResult` | `str()` / `os.fspath()` / `open()` / `os.path.exists()` 仍可用；`== "path"` 不再成立 |
| `ONNXExporter.verify(...)` | `bool` | `VerifyReport` | `__bool__` 返回 `passed`，`if` / `assert` / `not` 语义不变；`is True` / `== True` 不再成立 |
| `ONNXExporter(...)` 构造 | `model, input_shape, input_shapes, dummy_inputs, device, task_type, normalize` | 同上，**新增可选** `preprocess=None` | 旧关键字调用完全不变 |
| `from dvisionix.export import ONNXExporter` | 可用 | 仍可用 | 未改导入路径 |

典型迁移：

```python
# P5 之前
path = exporter.export("m.onnx")
ok = exporter.verify(path)

# P5 之后（两种写法都能跑；verify 也接受 ExportResult）
result = exporter.export("m.onnx")
report = exporter.verify(result)
print(report.passed, report.max_abs_diff, report.outputs)
```

注意点：

- `verify()` 传入 `ExportResult` 时会把全局最大误差回填到
  `result.parity_max_abs_diff`；传入路径字符串时没有对象可回填；
- `VerifyReport` 是 dataclass，`repr` 会打印完整明细，日志里不会退化成 `True`/`False`；
- `ExportResult.to_dict()` 中的 `dynamic_axes` 轴键在 JSON 里是字符串，`from_dict`
  会转回 `int`，不要自己手工 `json.load` 后直接比较。

## 5. Python API

### 5.1 单输入、多输入与自定义输入

```python
from dvisionix.export import ONNXExporter

# 单输入：默认输出名 "output"；input_shape 不含 batch 维
exporter = ONNXExporter(model, input_shape=(3, 224, 224))
exporter.export("cls.onnx")

# 多输入：input_shapes / dummy_inputs；默认输入名 input_0 / input_1 ...
exporter = ONNXExporter(model, input_shapes=[(3, 224, 224), (8,)])
exporter.export("multi.onnx", input_names=["image", "feature"])

exporter = ONNXExporter(model, dummy_inputs=(torch.randn(2, 8), torch.randn(2, 4)))
exporter.export("custom.onnx", dynamic_batch=False)
```

导出器**不会**在构造期修改调用者的模型：`export()` / `verify()` 只在
`_temporary_eval()` 内临时切到目标设备的 eval 模式，并在 `finally` 中恢复原设备与
`train`/`eval` 状态（失败路径同样恢复）。

### 5.2 输出结构：dict / list / 任意嵌套

`_flatten_outputs` 递归展平模型输出，输出名由访问路径生成：

```python
# {"logits": t, "feat": t2} → ["logits", "feat"]
# {"boxes": [t0, t1], "aux": {"cls": t2}}
#   → ["boxes_0", "boxes_1", "aux_cls"]
#     output_path_map = {"boxes_0": "boxes[0]", ..., "aux_cls": "aux.cls"}
```

支持的叶子类型只有 `Tensor`；`dict` / `list` / `tuple` 可任意嵌套，其它类型抛
`TypeError` 并指出出错路径。`output_names` 显式传入时数量必须与展平后的输出数一致，
否则抛 `ValueError`（不会静默截断）。

检测模型导出的是 **forward 的原始预测张量**（不含 decode / NMS），后处理需要调用方
在推理端自行实现，详见 [目标检测](detection.md)。

### 5.3 trace 与 dynamo 后端

```python
# trace（默认）：TorchScript 路径，零额外依赖；dynamic_batch 默认 True
result = exporter.export("m.onnx", backend="trace", dynamic_batch=True)

# dynamo：torch.export 新路径，需要 onnxscript；dynamic_batch 默认 False
result = exporter.export("m.onnx", backend="dynamo")
```

| 选项 | trace（默认） | dynamo |
|---|---|---|
| 额外依赖 | 无 | `pip install onnxscript` |
| 动态维度 | `dynamic_axes`（`dynamic_batch` / `dynamic_size`） | 用 PyTorch 的 `dynamic_shapes`（本实现未暴露该参数），**不支持** `dynamic_axes` |
| `dynamic_batch` 默认值 | `True`（历史行为） | `False` |
| 显式要求动态维度 | 支持 | 抛 `ValueError` 并说明原因 |
| 适用场景 | 常规 CNN / 检测 / 分割 | 含控制流或动态 shape 的自定义模型 |

`dynamic_batch` / `dynamic_size` 默认值是 `None` 哨兵：`trace` 下取历史默认
（`True` / `False`），`dynamo` 下取 `False` / `False`。因此最自然的
`exporter.export("m.onnx", backend="dynamo")` 可以直接用；只有**显式**
`dynamic_batch=True` 时才会报错，避免静默丢掉动态 batch 的承诺。

`dynamic_size=True` 目前只在 trace 后端生效：对每个秩 ≥ 3 的图像输入，把 axis 2 / 3
标记为 `height` / `width`（要求模型本身对 H/W 不敏感）。

### 5.4 `preprocess` 的推断规则

`ExportResult.preprocess` 的优先级：

1. 显式传入的 `preprocess=PreprocessSpec(...)`；
2. 未显式传入、但 `task_type` 非空：按 `self.normalize` 与输入形状推断——
   `image_size = input_shapes[0][-2]`，`mean` / `std` 取 `normalize`（缺省 ImageNet），
   `scale = normalize.get("scale", 1/255)`；
3. 两者都没有：`None`。

```python
from dvisionix.inference import PreprocessSpec

spec = PreprocessSpec(task_type="detection", image_size=640)   # 显式规格优先
exporter = ONNXExporter(model, input_shape=(3, 640, 640), preprocess=spec)
assert exporter.export("m.onnx").preprocess is spec
```

若设置了 `task_type` 但第一个输入不是图像（不足 2 维），推断会抛 `ValueError`，提示
该任务需要图像输入。CLI 总是把配置里的 `task_type` 与 `normalize` 传给导出器，因此
通过 CLI 导出的 sidecar 一般都有 `preprocess`（配置缺 `task_type` 时除外）。

### 5.5 精确控制导出

```python
result = exporter.export(
    "model.onnx", dynamic_batch=True, dynamic_size=False, opset_version=17,
    input_names=["image"], output_names=["logits"],  # 多输入/多输出需显式命名
    simplify=True, backend="trace",                  # simplify 需 onnxsim
    metadata={"experiment": "demo", "epoch": 3},     # 追加到 metadata_props
)
```

`simplify=True` 在导出、写 metadata 之后执行；`onnxsim` 未安装或简化校验失败时只告警，
保留原始计算图。`result.extra["simplify"]` 记录的是调用参数，不代表一定简化成功。

## 6. ONNX metadata

导出完成后，`ONNXExporter._write_metadata` 会向 ONNX 模型的 `metadata_props` 写入：

| 键 | 值 | 说明 |
|---|---|---|
| `dvisionix_export` | `json.dumps(result.to_dict())` | **P5 新增**：完整导出结果（路径 / 输入输出 / opset / 动态轴 / preprocess / extra） |
| `normalize_mean` | JSON 字符串，如 `[0.485, 0.456, 0.406]` | 沿用旧键，归一化均值 |
| `normalize_std` | JSON 字符串，如 `[0.229, 0.224, 0.225]` | 沿用旧键，归一化标准差 |
| `normalize_scale` | JSON 字符串，如 `0.00392156862745098` | 沿用旧键（`normalize` 含 `scale` 时） |
| `output_path_map` | JSON 字符串，如 `{"output": "output"}` | 输出名 → 模型输出访问路径 |
| 其它 | `metadata` 参数原样写入（非字符串值会 `json.dumps`） | 如 `experiment` / `epoch` |

读取方式：

```python
import json, onnx

props = {p.key: p.value for p in onnx.load("exports/cls.onnx").metadata_props}
info = json.loads(props["dvisionix_export"])
print(info["input_shapes"], info["preprocess"]["image_size"])
print(json.loads(props["output_path_map"]))    # {'output': 'output'}
```

注意：

- `normalize_*` 的值是**字符串**（内容是 JSON），读取时要 `json.loads`；字符串型
  `metadata` 不会被二次编码；
- `dvisionix_export` 记录的是**导出当时**的结果，`parity_max_abs_diff` 通常为 `null`；
  校验误差只回填到 Python 对象与 sidecar，**不会回写 ONNX 文件**；
- `onnx` 未安装时跳过 metadata 写入并告警（metadata 是可选增强信息）。

## 7. ONNX Runtime 一致性校验

`ONNXExporter.verify(result_or_path, rtol=1e-3, atol=1e-5, num_samples=3)` 的行为：

| 行为 | 说明 |
|---|---|
| 接受对象 | `ExportResult` 或路径字符串 |
| 执行后端 | `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])` |
| 输入 | 复用导出时的 `self.dummy_inputs`；每轮 `clone()`，共 `num_samples` 轮 |
| 比较 | 逐输出 `np.allclose(torch_array, onnx_out, rtol=rtol, atol=atol)` |
| 输出数不一致 | 直接抛 `ValueError`（`zip` 会静默截断，这里不放过） |
| 形状不一致 | 该输出记为未通过并记录 `[FAIL]` 日志，继续校验其它输出 |
| 汇总 | `report.max_abs_diff` 为**全局**最大绝对误差；`report.outputs` 为逐输出明细（跨样本取最大） |
| `passed` | 全部输出、全部样本都通过才为 `True` |
| 回填 | 传入 `ExportResult` 时写回 `result.parity_max_abs_diff` |
| 缺 onnxruntime | 告警并返回 `VerifyReport(passed=False)`（`max_abs_diff=None`、`outputs=[]`）——**是未通过，不是跳过** |

```python
report = exporter.verify("exports/cls.onnx", rtol=1e-3, atol=1e-5, num_samples=3)
print(report.passed, report.max_abs_diff)
for item in report.outputs:
    print(item["name"], item["shape"], item["max_abs_diff"], item["passed"])
print(report.to_dict())
```

`VerifyReport.to_dict()` 返回 `{"passed","max_abs_diff","outputs"}`，`outputs` 每项含 `name` / `shape` / `max_abs_diff` / `passed`。

容差约定：`rtol=1e-3` / `atol=1e-5` 是默认比较阈值，实际验收更关注
**全局 `max_abs_diff` < 1e-4**（本仓库三条真实链路的实测值均在 `1e-5` 量级及以下，
见第 11 节）。若自行调大 `atol`，请同时记录理由——`verify` 通过只说明数值一致性，
不代表模型效果。

> `verify` 用的是导出时的随机 dummy 输入，不是数据集级精度评估；它只回答
>「ONNX 与 PyTorch 在同一输入上是否一致」。

## 8. TensorRT 扩展点

`dvisionix/export/tensorrt.py` 只提供**可注册、可构建、可被探测**的扩展点：

```python
from dvisionix.export import EXPORTERS, TensorRTExporter, build_exporter

assert "tensorrt" in EXPORTERS
exporter = build_exporter("tensorrt", model)      # 可以构建
assert isinstance(exporter, TensorRTExporter)

exporter.export("model.trt")
# NotImplementedError: TensorRT 导出尚未实现：本类仅作为扩展点占位（P5 只交付 ONNX）。
```

- `TensorRTExporter.format == "tensorrt"`，继承 `BaseExporter`；
- `export()` 恒抛 `NotImplementedError`，避免调用方误以为已经支持；
- `verify()` 未覆盖，走 `BaseExporter.verify` 的默认实现，抛
  `NotImplementedError("该导出器未实现 verify")`；
- CLI `--format tensorrt` 能通过注册表检查，但会在 `export()` 处失败并非零退出。

P5 **没有**交付 TensorRT 引擎构建、校准、序列化或推理。需要 TensorRT 时，请先导出
ONNX，再用 NVIDIA 的 `trtexec` / TensorRT Python API 在仓库外自行转换（这属于外部
流程，本项目不做保证）。

## 9. 扩展新的导出格式

新格式只需三件事：继承 `BaseExporter`、实现 `export()`、注册到 `EXPORTERS`。

```python
from dvisionix.export import EXPORTERS, BaseExporter, ExportResult

@EXPORTERS.register(name="openvino")
class OpenVINOExporter(BaseExporter):
    format = "openvino"

    def export(self, output_path, **kwargs):
        # ... 真正的转换逻辑 ...
        return ExportResult(
            path=output_path, format=self.format,
            input_names=["input"], input_shapes=[[1, 3, 224, 224]],
            output_names=["output"], extra={"backend": "openvino"},
        )
```

注册后 `build_exporter("openvino", model, input_shape=(3, 224, 224))` 即可构建；
未注册的名字抛 `ConfigError`，错误消息会列出全部已注册格式（含刚注册的 `openvino`）。
若只想覆盖式注册（开发期重载），`Registry.register` 支持 `force=True`。

## 10. 已知限制

- **TensorRT 未实现**：`TensorRTExporter` 只是扩展点，`export()` 恒抛
  `NotImplementedError`；没有引擎构建 / 校准 / 序列化。
- **未做量化 / 裁剪**：导出的是训练时的 float32 计算图；没有 INT8 / FP16 量化、没有
  剪枝、没有蒸馏后再导出，也没有校准数据集流程。
- **dynamo 后端**：需要额外安装 `onnxscript`；不支持 `dynamic_axes`，显式要求动态
  batch / size 会抛 `ValueError`；其默认 `dynamic_batch=False` 与 trace 相反。
- **trace 后端**：零额外依赖，但 PyTorch ≥ 2.9 会提示 TorchScript 导出已废弃（仍可用）；
  含复杂控制流 / 数据依赖分支的模型建议 dynamo。
- **`dynamic_size`**：仅 trace 后端支持，且要求模型对 H/W 变化不敏感；对固定尺寸的
  检测 / 分割模型不要打开。
- **simplify**：需要 `pip install onnxsim`；未安装或简化校验失败时只告警并保留原图。
- **检测输出不含后处理**：导出的是原始预测张量，decode / NMS / 阈值需要推理端实现。
- **CLI 只构造单个图像输入**：`--backend` / `--simplify` / `--input-names` /
  `--output-names` / `--dynamic-size` 未暴露；多输入 / 自定义 dummy 输入、多输出命名
  请用 Python API。sidecar 命名固定为 `<output>.json`（不是 `.onnx.json`）。
- **校验是单输入随机样本级**：`verify` 用导出时的 dummy 输入，不是数据集级精度评估；
  缺 onnxruntime 时返回 `passed=False`（CLI 退出码 1），是**未通过**而非跳过。
- **ONNX metadata 不回写**：`dvisionix_export` 里的 `parity_max_abs_diff` 保持导出时的
  `null`，校验结果只在 sidecar / Python 对象里。
- **多任务分类 / SimCLR 的导出未验证**：P3 的路由头需要每样本 `task`，CLI 只构造单张量
  输入；这两类模型的导出暂无端到端验收。

## 11. 测试与验收

导出模块的测试：

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python

$PY -m pytest tests/test_export -q
# 既有：test_onnx_export.py / test_onnx_contract.py（原样通过）
# P5：test_export_base.py / test_export_cli.py
```

全量测试基线：**1089 passed / 1 skipped / 0 failed**（v2.0.1）。

真实 checkpoint 验收（1 epoch 冒烟 checkpoint，**仅链路证据，不作为效果结论**）：

| 任务 | 配置 | 输入 | 输出 | 全局 `max_abs_diff` |
|---|---|---|---|---|
| 分类 | `configs/examples/dvx_classification.yaml` | `[1, 3, 224, 224]` | `[1, 5]` | **5.96e-06** |
| 检测 | `configs/examples/dvx_detection.yaml` | `[1, 3, 512, 512]` | 8 个：`cls_0..3` / `reg_0..3` | **3.34e-06** |
| 分割 | `configs/examples/dvx_segmentation.yaml` | `[1, 3, 256, 256]` | `[1, 18, 256, 256]` | **4.77e-06** |

三者均 < 1e-4；检测 8 个输出形状为 `cls_i: [1, 20, 128, 128] / [1, 20, 64, 64] /
[1, 20, 32, 32] / [1, 20, 16, 16]`、`reg_i: [1, 4, ...]`（20 类配置）。sidecar 的
`preprocess` / `input_shapes` 与上表一致，`--verify` 时 `parity_max_abs_diff` 已回填；
同一分类 checkpoint 复跑得到 `2.19e-05`（正常波动，量级仍是 **约 1e-5**）。

## 12. 版本变更要点（仍有效的行为）

> 完整历史见 [CHANGELOG](../CHANGELOG.md)。

- **任意嵌套输出（v1.1 起）**：递归展平 `dict` / `list` / `tuple`，输出名由访问路径
  生成（`boxes_0` / `aux_cls`），并写入 `metadata_props.output_path_map`；此前只处理一层。
- **不修改调用者模型（v1.1 起）**：只在 `export()` / `verify()` 期间临时切设备与 eval
  模式，`finally` 恢复原状态（失败路径同样恢复）。
- **verify 更严格（v1.1 起）**：统一 `detach().cpu().numpy()`；输出数量或形状不一致时
  显式报错或记失败，而不是被 `zip` 静默截断。
- **dynamo 默认调用可用（v1.1.1 起）**：`dynamic_batch` / `dynamic_size` 用 `None`
  哨兵，trace 取 `True` / `False`、dynamo 取 `False` / `False`，只有显式要求动态维度
  才报错。
- **P5 导出体系（当前）**：`BaseExporter` + `EXPORTERS` + `ExportResult` /
  `VerifyReport`，`tools/export.py` CLI 与 metadata 的 `dvisionix_export` 键。

## 13. 相关文档

- [快速开始](quick_start.md) —— 安装、训练、推理与一条导出命令
- [推理链路](inference.md) —— `PreprocessSpec` / `PreprocessPipeline` 与 `Results`
- [目标检测](detection.md) / [语义分割](segmentation.md) —— 检测输出、分割数据格式
- [P5 导出接口契约](history/design/v2_p5_contract.md) —— 实现契约（权威）
- [变更日志](../CHANGELOG.md) —— 版本历史
