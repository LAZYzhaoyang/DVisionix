# DVisionix v2 —— P5 导出接口契约（Lead 制定，实现方必须遵守）

> 状态：**生效**。P5 = 把"只能导出 ONNX"升级为**有接口、可注册、可校验、可扩展**的导出体系。
> 环境：`PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python`（onnx 1.23.2 / onnxruntime 1.31.0 / onnxscript 0.7.2 已装）。
> 风格：UTF-8；中文 docstring；ruff 100 / black 100；不新增依赖。

## 0. 目标与向后兼容

**目标**：
1. `BaseExporter` 抽象 + `EXPORTERS` 注册表 + `ExportResult`（含 `PreprocessSpec` 元数据）；
2. 现有 `ONNXExporter` 重构为 `BaseExporter` 实现，**`from dvisionix.export import ONNXExporter` 仍可用**；
3. 预留 `TensorRTExporter` 扩展点（注册 + 明确 NotImplementedError）；
4. `tools/export.py` CLI + ONNX Runtime 一致性校验。

**兼容策略（必须遵守）**：
- `ONNXExporter.export(...)` 返回值由 `str` 改为 `ExportResult`；`ExportResult` 实现
  `__str__` 与 `__fspath__`，使 `str(result)`/`open(result)`/`os.path.exists(result)` 依然工作；
  仓库内没有任何调用方依赖 `export() == "path"`（已核实），文档需注明该返回类型变化。
- `ONNXExporter.verify(...)` 由 `bool` 改为 `VerifyReport`；`VerifyReport.__bool__` 返回 `passed`，
  因此既有 `assert exporter.verify(path)` 语义不变（已核实测试写法）。
- 构造函数签名保持兼容，仅**新增**可选参数 `preprocess`。

## 1. `dvisionix/export/base.py`（新增）

```python
EXPORTERS = Registry("exporters")        # 复用 dvisionix.registry.Registry

@dataclasses.dataclass
class ExportResult:
    path: str
    format: str                          # "onnx" / "tensorrt" / ...
    input_names: list[str]
    input_shapes: list[list[int]]        # 每个 dummy 输入的完整形状（含 batch=1）
    output_names: list[str]
    opset: int | None = None
    dynamic_axes: dict[str, dict[int, str]] | None = None
    preprocess: "PreprocessSpec | None" = None    # from dvisionix.inference.preprocess
    parity_max_abs_diff: float | None = None      # verify 后回填
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict: ...       # preprocess → to_dict()
    @classmethod
    def from_dict(cls, data: dict) -> "ExportResult": ...
    def save(self, path: str) -> None: ...        # JSON sidecar（indent=2, ensure_ascii=False）
    @classmethod
    def load(cls, path: str) -> "ExportResult": ...
    def __str__(self) -> str: return self.path
    def __fspath__(self) -> str: return self.path

@dataclasses.dataclass
class VerifyReport:
    passed: bool
    max_abs_diff: float | None = None
    outputs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    def __bool__(self) -> bool: return self.passed
    def to_dict(self) -> dict: ...

class BaseExporter(ABC):
    format: ClassVar[str] = ""
    def __init__(self, model, **kwargs) -> None: self.model = model; self.options = dict(kwargs)
    @abstractmethod
    def export(self, output_path: str, **kwargs) -> ExportResult: ...
    def verify(self, result_or_path, **kwargs) -> VerifyReport:
        raise NotImplementedError("该导出器未实现 verify")

def build_exporter(format: str, model, **kwargs) -> BaseExporter:
    """按注册名构建导出器；未知格式抛 ConfigError（不是裸 KeyError）。"""
```

## 2. `ONNXExporter` 重构（`dvisionix/export/onnx_exporter.py`）

- 继承 `BaseExporter`，`format = "onnx"`，用 `@EXPORTERS.register(name="onnx")` 注册。
- `__init__` 保持现有参数与默认值，新增可选 `preprocess: PreprocessSpec | None = None`；
  `super().__init__(model)`。
- `export(...)`：内部逻辑不变，返回 `ExportResult`：
  - `path`、`format="onnx"`、`input_names`、`input_shapes=[list(t.shape) for t in dummy_inputs]`、
    `output_names`、`opset=opset_version`、`dynamic_axes=实际使用的 dynamic_axes（无则 None）`；
  - `preprocess`：显式传入优先；否则若 `self.task_type` 非空，用
    `PreprocessSpec(task_type=self.task_type, image_size=int(input_shapes[0][-2]),
    mean/std 来自 self.normalize（缺省 ImageNet）, scale=normalize.get("scale", 1/255))` 构造；都没有则 None；
  - `extra={"backend": backend, "simplify": simplify, "output_path_map": ...}`。
- 把 `result.to_dict()` 序列化后写入 ONNX `metadata_props` 的 `dvisionix_export` 键
  （保留原有 `normalize_*` 与 `output_path_map` 键不变）。
- `verify(result_or_path, rtol=1e-3, atol=1e-5, num_samples=3) -> VerifyReport`：
  接受 `ExportResult` 或路径；保留原比较逻辑；汇总**全局 max_abs_diff** 与逐输出明细
  `outputs=[{"name", "shape", "max_abs_diff", "passed"}]`；`passed = 全部输出所有样本通过`。

## 3. `TensorRTExporter` 扩展点（新增 `dvisionix/export/tensorrt.py`）

```python
@EXPORTERS.register(name="tensorrt")
class TensorRTExporter(BaseExporter):
    format = "tensorrt"
    def export(self, output_path: str, **kwargs) -> ExportResult:
        raise NotImplementedError(
            "TensorRT 导出尚未实现：本类仅作为扩展点占位（P5 只交付 ONNX）。"
        )
```
（注册必须成功；`export` 必须抛 `NotImplementedError`，测试覆盖。）

## 4. `dvisionix/export/__init__.py`

导出（保留旧名）：`ONNXExporter`、`BaseExporter`、`ExportResult`、`VerifyReport`、`EXPORTERS`、
`build_exporter`、`TensorRTExporter`；`__all__` 完整。

## 5. `tools/export.py`（新增 CLI）

```
python tools/export.py --config <yaml> --checkpoint <pt> --output <model.onnx>
  [--format onnx] [--dynamic-batch] [--opset 17] [--verify] [--trusted]
  [--device cpu] [--cfg-options k=v ...]
```
行为：
1. `Config.from_yaml` + `update_from_cli`；`build_model(cfg.model.to_dict())`。
2. 加载 checkpoint（`torch.load(map_location="cpu", weights_only=not trusted)`；取 `model_state_dict`
   或整体；`strip_wrapper_prefixes`；`load_state_dict(strict=True)`）；失败抛 `CheckpointError` 并提示 `--trusted`。
3. 输入形状：`in_channels = cfg.model.get("in_channels", 3)`，`image_size = cfg.data.get("image_size", 32)`；
   normalize：从 `cfg.data.get("mean"/"std", ImageNet 默认)` 构造 dict（含 `scale=1/255`）。
4. `build_exporter(args.format, model, input_shape=(C,H,W), device=args.device,
   task_type=cfg.get("task_type"), normalize=normalize)` → `export(output, dynamic_batch=..., opset_version=...)`。
5. `result.save(output + ".json")` 落盘 sidecar；打印 path/输入输出名/`preprocess` 摘要。
6. `--verify`：`report = exporter.verify(result)`；打印 `max_abs_diff` 与逐输出结果；
   未通过时以非零码退出（`sys.exit(1)`）。
7. `--trusted` 语义与推理 CLI 一致。

## 6. 测试

- `tests/test_export/test_export_base.py`（A 写）：
  - `ExportResult` 的 `to_dict/from_dict/save/load` 往返（含 `PreprocessSpec`）；
    `str(result) == result.path`、`os.fspath(result) == path`、`open(result)` 可读；
  - `VerifyReport.__bool__`；
  - `EXPORTERS` 中 `onnx`/`tensorrt` 均已注册；`build_exporter("onnx", model)` 返回 `ONNXExporter`
    且 `isinstance(..., BaseExporter)`；未知格式抛 `ConfigError`；
  - `TensorRTExporter.export` 抛 `NotImplementedError`；
  - 导出后 `result.preprocess` 非空且 `task_type`/`image_size` 正确；ONNX metadata 含 `dvisionix_export` 键。
- `tests/test_export/test_export_cli.py`（B 写）：`tmp_path` 造最小 config + `SimpleCNN(num_classes=3)` 纯
  state_dict checkpoint → 调 `tools.export.main()` → 生成 `.onnx` 与 `.onnx.json`；`--verify` 通过；
  不带 `--trusted` 加载完整 Trainer checkpoint 时抛 `CheckpointError`。
- **既有 `tests/test_export/{test_onnx_contract.py,test_onnx_export.py}` 必须原样通过**（可加不可改）。

## 7. 验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python
$PY -m pytest tests/test_export -q
$PY -m pytest tests -q                     # 0 failed（基线 1066 passed / 1 skipped）
$PY -m ruff check dvisionix tools tests && $PY -m black --check dvisionix tools tests
$PY tools/check_layering.py
```
Lead 追加：真实 cls/det/seg checkpoint 各导出 ONNX + `--verify` 通过（ORT 与 PyTorch max_abs_diff < 1e-4），
并检查 sidecar JSON 中 `preprocess` 与 `input_shapes` 正确。

## 8. 文件所有权

| 归属 | 文件 |
|---|---|
| A（导出核心） | `dvisionix/export/{base.py(新),onnx_exporter.py,tensorrt.py(新),__init__.py}`、`tests/test_export/test_export_base.py` |
| B（CLI） | `tools/export.py`、`tests/test_export/test_export_cli.py` |
| Lead | 本契约、审查、真实数据导出验收、git 提交；文档 Wave 另派 |
