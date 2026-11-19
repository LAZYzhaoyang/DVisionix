# DVisionix v2 —— P4 推理链路契约（Lead 制定，实现方必须遵守）

> 状态：**生效**。P4 = 训练好的模型 → 预处理 → 前向 → 后处理/反变换 → 统一结果 → JSON/可视化 + CLI。
> 环境：`PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python`；CUDA 可用。
> 风格：UTF-8；中文 docstring；ruff 100 / black 100；不得新增第三方依赖（只用 torch/numpy/cv2/PIL）。

## 0. 目标与非目标

**目标**：补齐"训练 → 能用"之间缺失的主链：
- 预处理：与训练/验证**完全一致**的确定性流水线（复用现有 `*Transforms(train=False)`，不重写 resize/normalize）；
- 后处理：分类 softmax/top-k、检测复用 `model.decode()`、分割 argmax；
- **反变换**：框/掩码映射回**原图坐标与原始尺寸**；
- 统一 `Results` 对象 + JSON 落盘 + 可视化；
- CLI `tools/infer.py`。

**非目标**：ONNX/TRT 导出（P5）、批量吞吐优化、视频流、服务化。

## 1. `dvisionix/inference/preprocess.py`（A1）

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclasses.dataclass(frozen=True)
class PreprocessSpec:
    task_type: str
    image_size: int
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    scale: float = 1.0 / 255.0

    @classmethod
    def from_pipeline(cls, pipeline, task_type: str = "") -> "PreprocessSpec": ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessSpec": ...

@dataclasses.dataclass
class PreprocessMeta:
    original_size: tuple[int, int]     # (H, W)
    processed_size: tuple[int, int]    # (H, W)
    scale: tuple[float, float]         # (sh, sw) = 处理后 / 原图
    image_path: str | None = None
    def to_dict(self) -> dict: ...

class PreprocessPipeline:
    def __init__(self, spec: PreprocessSpec, pipeline=None) -> None: ...
    @classmethod
    def from_task(cls, task_type: str, image_size: int) -> "PreprocessPipeline": ...
    @property
    def spec(self) -> PreprocessSpec: ...
    def __call__(self, image) -> tuple["torch.Tensor", PreprocessMeta]: ...
```

规则：
1. `from_pipeline`：遍历 `pipeline.transforms` 提取 `image_size`（取第一个有 `.size` 的变换的 `size[0]`）与
   `ImageNormalize` 的 `mean/std/scale`（numpy 数组→float tuple）；缺失时用 ImageNet 默认。
2. `PreprocessPipeline.from_task`：用 `dvisionix.training.tasks` 中对应 Task 类的
   `build_transforms(train=False, image_size=...)` 构建（**单一事实来源**，禁止另写一份变换）。
   任务映射：classification/multi_task_classification→ClassificationTask；detection→DetectionTask；
   segmentation/MaskFormer→SegmentationTask；simclr→SimCLRTask。
3. `__call__` 输入支持：`str`/`Path`（cv2 读取并 BGR→RGB）、`np.ndarray`（uint8 HWC，**按 RGB 解释**）、
   `PIL.Image`；其它类型抛 `InferenceError`。返回 `(1, 3, H, W) float32 tensor`（CPU）与 `PreprocessMeta`。
4. 复用验证 pipeline：构造 `Sample({"image": arr, "meta": {}})` 后跑 pipeline；从 `sample["meta"]` 读取
   `original_size` / `scale`（检测/分割预设会写），缺失则按"原图尺寸 vs 处理后尺寸"计算；
   `processed_size` 由输出张量形状推导。
5. **确定性**：同一输入重复调用结果完全一致（有测试）；不修改传入的 ndarray。

## 2. `dvisionix/inference/results.py`（A1）

```python
@dataclasses.dataclass
class Results:
    task_type: str
    orig_size: tuple[int, int]                 # (H, W)
    image_path: str | None = None
    # 分类
    probs: np.ndarray | None = None            # (C,)
    topk: list[tuple[int, float]] | None = None
    # 检测
    boxes: np.ndarray | None = None            # (N,4) xyxy **原图坐标**
    scores: np.ndarray | None = None
    labels: np.ndarray | None = None
    # 分割
    mask: np.ndarray | None = None             # (H,W) int64 **原图尺寸**
    meta: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict: ...             # numpy→list；不含原始图像
    def to_json(self, path: str | None = None) -> str: ...   # indent=2, ensure_ascii=False
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Results": ...
    def visualize(self, image=None, class_names=None) -> np.ndarray: ...  # 返回 BGR 图像
```
`visualize`：分类→左上角 top-k 文本；检测→框+`label score`；分割→调色板叠色 0.5；
`image` 为 None 且有 `image_path` 时读图，否则在深色底上画。

## 3. `dvisionix/inference/predictors.py`（A2）

```python
class BasePredictor:
    def __init__(self, model, task_type: str, device="auto", image_size: int = 224,
                 class_names: list[str] | None = None,
                 preprocess: PreprocessPipeline | None = None, **kwargs) -> None: ...
    @classmethod
    def from_checkpoint(cls, model, checkpoint: str, task_type: str, *,
                        trusted: bool = False, **kwargs) -> "BasePredictor": ...
    def predict(self, image) -> Results: ...
    def predict_many(self, inputs: Sequence) -> list[Results]: ...

class ClassificationPredictor(BasePredictor):    # 额外参数 topk: int = 5
class DetectionPredictor(BasePredictor):         # score_threshold=0.3, iou_threshold=0.5, max_detections=100
class SegmentationPredictor(BasePredictor):      # 额外参数 return_probs: bool = False

def build_predictor(model, task_type: str, **kwargs) -> BasePredictor: ...
```

规则：
1. `from_checkpoint`：`torch.load(map_location="cpu", weights_only=not trusted)`；
   取 `model_state_dict`（无该键则把整个对象当 state_dict）；`strip_wrapper_prefixes`；
   `model.load_state_dict(strict=True)`。失败抛 `CheckpointError`，hint 说明
   "完整 Trainer checkpoint 含非张量状态，需 trusted=True"。
2. `predict`：临时 `model.eval()` + `torch.no_grad()`，结束后**恢复**原 training 状态；
   预处理 → 张量搬到 `self.device`；
   forward 前向（`getattr(model, "needs_batch", False)` 时传 `batch={"image": images}`，与 DetectionTask 一致）；
   按任务后处理：
   - 分类：`softmax(logits[0])` → top-k（k=min(topk,C)）→ `Results(probs, topk)`；
   - 检测：`model.decode(preds, processed_hw, score_threshold, iou_threshold, max_detections)`
     → 取 batch0 的 (boxes, scores, labels) → **反变换**：`x / sw`, `y / sh`，
     clip 到 `[0, W-1]/[0, H-1]` → `Results`；
   - 分割：`logits.argmax(1)` → `F.interpolate(..., size=orig_size, mode="nearest")`
     → `(H,W) int64` → `Results`（`return_probs=True` 时附带 softmax 概率到 `Results.meta`）。
3. `predict_many`：逐个 `predict`，返回列表（v1 不做跨图 batch 堆叠）。
4. `build_predictor` 按 `task_type` 返回对应子类；不支持的类型抛 `ConfigError`。

## 4. `tools/infer.py`（B）

```
python tools/infer.py --config <训练/示例配置 yaml> --checkpoint <pt> --input <图片或目录> --output <目录>
  [--task-type X] [--device auto] [--trusted] [--save-json/--no-save-json] [--save-vis]
  [--cfg-options k=v ...]
```
- 从 config 用 `build_model(cfg.model.to_dict())` 构建模型；`task_type` 缺省取 `cfg.task_type`；
  `image_size` 取 `cfg.data.image_size`；`class_names` 若 config 有 `data.classes` 则用之。
- input 为目录：递归收集 `.jpg/.jpeg/.png/.bmp`（排序）；为文件则单个。
- 每个输入输出 `<stem>.json`；`--save-vis` 时再输出 `<stem>_vis.jpg`。
- stdout 打印汇总（处理数量、耗时）；任何失败非零退出。

## 5. 其他改动

- `dvisionix/errors.py`（A1）：新增 `class InferenceError(DVisionixError)`。
- `dvisionix/__init__.py`（B）：`_SUBMODULES` 与 `__all__` 增加 `"inference"`（保持懒加载）。
- `dvisionix/inference/__init__.py`（A1）：导出 `PreprocessSpec/PreprocessMeta/PreprocessPipeline/Results`，
  并为 `predictors` 保留 re-export（A2 完成后由 A2 补充，或 A1 先占位）。

## 6. 测试

- `tests/test_inference/__init__.py`
- `test_preprocess.py`（A1）：三种 pipeline 的 spec 提取（cls/det/seg）、路径/ndarray/PIL 输入、
  meta 的 original_size/scale、确定性、非法类型抛 `InferenceError`。
- `test_results.py`（A1）：`to_dict/to_json/load` 往返、三种任务 `visualize` 输出形状与 dtype。
- `test_predictors.py`（A2）：toy 三模型（`SimpleCNN(num_classes=3)`、`GridDetectionModel(num_classes=3)`、
  `SimpleSegmentationModel(num_classes=3)`）端到端；检测框全部落在原图范围内；分割 mask 形状=原图；
  分类 probs 和为 1、topk 有序；`from_checkpoint` 正常往返 + 损坏 checkpoint 抛 `CheckpointError`；
  预测后模型 training 状态被恢复。
- `test_infer_cli.py`（B）：tmp 目录放 2 张图 + 用 toy 模型保存一个 checkpoint + 最小 config，
  跑 CLI 主函数，断言输出 2 个 json（`--save-vis` 时另有 jpg）。

## 7. 验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python
$PY -m pytest tests/test_inference -q
$PY -m pytest tests -q                       # 0 failed（基线 963 passed / 1 skipped）
$PY -m ruff check dvisionix tools tests && $PY -m black --check dvisionix tools tests
$PY tools/check_layering.py
```
Lead 追加验收：用真实数据训练 1 epoch 得到 checkpoint，再用 `tools/infer.py` 对真实图片推理，
检查 JSON 与可视化产物（cls/det/seg 三类）。

## 8. 文件所有权

| 归属 | 文件 |
|---|---|
| A1 | `dvisionix/inference/{__init__.py,preprocess.py,results.py}`、`dvisionix/errors.py`、`tests/test_inference/{__init__.py,test_preprocess.py,test_results.py}` |
| A2 | `dvisionix/inference/predictors.py`、`tests/test_inference/test_predictors.py`、`dvisionix/inference/__init__.py`（仅追加 predictors 相关导出） |
| B | `tools/infer.py`、`dvisionix/__init__.py`、`tests/test_inference/test_infer_cli.py` |
| Lead | 本契约、审查、真实数据验收、git 提交；文档 Wave 另派 |
