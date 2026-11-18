# 推理（Inference）

> 适用范围：v2.0.0（P4 推理链路）。本页面向使用者，说明如何用训练好的
> checkpoint 做分类 / 检测 / 分割推理，以及预处理、统一结果对象与反变换的约定。
>
> 接口定义以实现契约 [P4 推理链路契约](design/v2_p4_contract.md) 为准，本页只描述实际行为。
> 相关文档：[快速开始](quick_start.md)、[训练契约与自适应训练](training_contracts.md)、
> [目标检测](detection.md)、[语义分割](segmentation.md)、[模型导出](model_export.md)。

---

## 1. 概念与数据流

P4 补齐的是「训练完 → 真正能用」之间缺失的主链：

```text
原图 (H, W)                 PreprocessPipeline                    模型
jpg/png/bmp ──读取──► resize + normalize + ToTensor ──► (1, 3, H', W') ──► forward
                                                                          │
                                    ┌─────────────────────────────────────┘
                                    ▼
                         任务后处理 + 反变换
          分类：softmax → top-k
          检测：model.decode → x/sw、y/sh → clip
          分割：argmax → nearest 插值回 (H, W)
                                    │
                                    ▼
              Results（原图坐标 / 原尺寸；to_json / visualize）
```

三个硬约定：

1. **预处理与验证期完全一致**。`PreprocessPipeline.from_task()` 不自己实现 resize 或
   normalize，而是调用 `dvisionix.training.tasks` 中对应 Task 的
   `build_transforms(train=False, image_size=...)`。训练/验证用哪套变换，推理就用哪套；
   预设改动只有一处，不会出现「训练/推理两套实现悄悄漂移」。
2. **所有几何量都已回到原图**。`Results.boxes` 是原图坐标的 xyxy，`Results.mask` 是原图
   `(H, W)`，`Results.orig_size` 是原图 `(H, W)`；调用方不需要再关心预处理缩放。
3. **一次一张图**。`predict()` 处理单张图；`predict_many()` 只是逐个调用 `predict()`
   的便捷方法，当前不做跨图 batch 堆叠（不同原图尺寸无法直接拼成一个张量）。

### 1.1 预处理（`PreprocessPipeline`）

`dvisionix/inference/preprocess.py` 提供：

- `PreprocessSpec`：预处理规格（任务类型、`image_size`、`mean/std/scale`），不可变；
- `PreprocessMeta`：单次预处理的元信息（`original_size` / `processed_size` / `scale` /
  `image_path`），反变换所需的信息都在这里；
- `PreprocessPipeline`：把原图变成 `(1, 3, H, W)` float32 CPU 张量 + `PreprocessMeta`。

`__call__` 支持的输入：

| 输入 | 约定 |
|---|---|
| `str` / `pathlib.Path` | 用 OpenCV 读取，BGR→RGB；`image_path` 写入 meta |
| `np.ndarray` | 必须是 `uint8` HWC，**按 RGB 解释**；`(H,W)` / 单通道会复制成 3 通道，4 通道丢弃 alpha |
| `PIL.Image.Image` | 先 `convert("RGB")` |

其它类型或非法形状/dtype 会抛 `InferenceError`。流水线每次使用全新的 `Sample`，同一输入
重复调用结果完全一致，且不会修改调用方传入的 ndarray（内部做防御性拷贝）。

### 1.2 前向与后处理

`BasePredictor` 负责设备解析、预处理、`eval()` + `torch.no_grad()` 前向和训练状态恢复；
具体任务子类只做任务语义的后处理：

- 分类：`softmax(logits[0])` → top-k（`k = min(topk, C)`）；
- 检测：调用模型自己的 `decode()`（阈值筛选 / NMS 与验证步同一契约），再把框反变换回
  原图坐标并裁剪；
- 分割：`argmax(1)` → `nearest` 插值回原图尺寸，得到 `(H, W) int64` 类别图。

> 检测的阈值筛选与 NMS **不在预测器里重写**，而是全部交给 `model.decode()`。自定义检测器
> 只要实现统一的 `decode(preds, image_hw, score_threshold=..., iou_threshold=...,
> max_detections=...)` 契约，就能直接接入 `DetectionPredictor`。

### 1.3 颜色约定（容易踩）

预处理和可视化的 numpy 颜色约定**相反**，这是有意的：

- `PreprocessPipeline` / `predict`：numpy 数组按 **RGB** 解释（与训练数据一致）；
- `Results.visualize`：numpy 底图按 **BGR** 解释（与 OpenCV 一致，返回值也是 BGR）。

路径输入两边都由 OpenCV 读取并各自转换，不需要调用方额外处理。

---

## 2. 快速开始（CLI）

### 2.1 准备 checkpoint

`tools/train.py` 默认把 checkpoint 写到工作目录的 `checkpoints/` 下：

```text
<work_dir>/checkpoints/best.pt     # 按监控指标保存的最优权重
<work_dir>/checkpoints/last.pt     # 最近一个 epoch
<work_dir>/ema_last.pt             # 开启 EMA 且 save_final=true 时的纯权重
```

`best.pt` / `last.pt` 是 **完整 Trainer checkpoint**（含优化器 / 调度器 / RNG / numpy
随机状态等），默认的安全加载模式无法读取，CLI 需要显式加 `--trusted`；`ema_last.pt`
是纯权重字典，默认即可加载（原因见 §2.3）。

### 2.2 三任务命令

`tools/infer.py` 的四个必填参数是 `--config` / `--checkpoint` / `--input` / `--output`，
其余按需：

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python
WORK=/path/to/训练输出目录          # 即训练日志里的 work_dir，按任务替换

# 分类
$PY tools/infer.py --config configs/classification/demo_synthetic.yaml \
  --checkpoint "$WORK/checkpoints/best.pt" --input ./samples/cls \
  --output ./outputs/cls --trusted --save-vis
# 检测
$PY tools/infer.py --config configs/detection/demo_synthetic.yaml \
  --checkpoint "$WORK/checkpoints/best.pt" --input ./samples/det \
  --output ./outputs/det --trusted --save-vis
# 分割
$PY tools/infer.py --config configs/segmentation/demo_synthetic.yaml \
  --checkpoint "$WORK/checkpoints/best.pt" --input ./samples/seg \
  --output ./outputs/seg --trusted --save-vis
```

> `--config` 必须与训练时使用的一致：模型结构、`num_classes` 和 `data.image_size`
> 都从配置读取；对不上 checkpoint 时 `load_state_dict` 会抛 `CheckpointError`。

### 2.3 `--trusted` 为什么必要

加载权重时默认走安全模式（`torch.load(..., weights_only=True)`）。`weights_only=True`
只允许反序列化张量和基本容器。完整 Trainer checkpoint 还会带上优化器状态、RNG、
numpy 随机状态等；其中 numpy RNG 等非张量对象会被安全模式拒绝。此时预测器会把底层
`UnpicklingError` 包装成 `CheckpointError`，实际输出形如（长文本已截断）：

```text
dvisionix.errors.CheckpointError: 加载 checkpoint 失败：.../checkpoints/best.pt
（UnpicklingError: Weights only load failed. This file can still be loaded, to do so you
have two options, do those steps only if you trust the source of the checkpoint.
...
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct
was not an allowed global by default. ...
）
提示：完整 Trainer checkpoint 含非张量状态（优化器 / RNG / numpy 等），安全模式
（weights_only=True）无法加载，需要 trusted=True；如果确认文件来源可信仍失败，
说明文件已损坏或不完整，请重新训练/导出。
```

解决方式就是给 CLI 加 `--trusted`（Python API 对应 `trusted=True`）。注意：

- `--trusted` 等价于 `torch.load(..., weights_only=False)`，反序列化 pickle 理论上可执行
  任意代码。**只对自己训练/确认来源的 checkpoint 使用**；不要对下载来的未知 `.pt` 使用。
- 纯 `state_dict`（只含张量的 `.pt`，例如 EMA 导出的 `ema_last.pt`）默认即可加载，不需要
  `--trusted`；如果模型权重是自己重新保存的，优先用这种形态。
- 即使加了 `--trusted`，如果文件损坏或不完整，仍会抛 `CheckpointError`（提示重新训练/导出）。

### 2.4 输入、输出与常用选项

- `--input` 可以是单张图片，也可以是目录；目录会**递归**收集
  `.jpg` / `.jpeg` / `.png` / `.bmp`（大小写不敏感），并按路径排序，保证结果可复现。
- 每张图输出 `<stem>.json`；`--save-vis` 时额外输出 `<stem>_vis.jpg`。
- 默认保存 JSON；不需要时加 `--no-save-json`（`--save-json` 是默认行为）。
- `--task-type` 缺省取配置里的 `task_type`，可显式覆盖；`--device` 默认 `auto`
  （CUDA → MPS → CPU），也可用 `cpu` / `cuda` / `cuda:1`。
- `--cfg-options` 可覆盖配置，例如 `--cfg-options data.image_size=224`。**改尺寸必须与
  训练时一致**，否则预处理的分布和模型期望会错位。
- `--config` 中 `data.classes` 存在时会作为类别名传给预测器：JSON 的
  `meta["class_names"]` 和可视化标签都会使用它。
- stdout 每张图打印 `[i/N] path`，结束时打印任务类型、图片数量、耗时和输出目录；
  任何失败都以非零码退出（异常不会被吞掉）。

---

## 3. Python API

### 3.1 `build_predictor` + `Results`

```python
import cv2
from dvisionix.config import Config
from dvisionix.inference import build_predictor
from dvisionix.models import build_model

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())        # 结构必须与 checkpoint 一致
predictor = build_predictor(
    model, "classification", device="auto",
    image_size=cfg.data.image_size,              # 必须与训练一致
    class_names=cfg.data.get("classes"), topk=5,
)
# 权重加载见 §3.2；这里假设 model 已载入训练好的权重，否则输出来自随机权重。

results = predictor.predict("cat.jpg")
print(results.task_type, results.orig_size, results.topk)
print(float(results.probs.sum()))                # 分类概率和为 1
results.save("outputs/cat.json")
vis = results.visualize(image="cat.jpg", class_names=results.meta.get("class_names"))
cv2.imwrite("outputs/cat_vis.jpg", vis)          # OpenCV 直接写 BGR
```

`build_predictor` 按 `task_type` 返回具体子类：

| `task_type` | 返回类型 | 额外常用参数 |
|---|---|---|
| `classification` | `ClassificationPredictor` | `topk=5` |
| `detection` | `DetectionPredictor` | `score_threshold=0.3`、`iou_threshold=0.5`、`max_detections=100` |
| `segmentation` | `SegmentationPredictor` | `return_probs=False` |

`task_type` 大小写和首尾空白不敏感；不支持的类型抛 `ConfigError`（见 §7）。

### 3.2 `BasePredictor.from_checkpoint`

`from_checkpoint` 定义在 `BasePredictor` 上，由三个任务子类继承。它读取 checkpoint
（纯 state_dict 也兼容）、剥离 DDP（`module.`）/ `torch.compile`（`_orig_mod.`）前缀，
并用 `strict=True` 载入模型。

```python
from dvisionix.config import Config
from dvisionix.inference import ClassificationPredictor
from dvisionix.models import build_model

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())
predictor = ClassificationPredictor.from_checkpoint(
    model, "checkpoints/best.pt", "classification",
    trusted=True,                                # 完整 Trainer checkpoint 必需
    device="auto", image_size=cfg.data.image_size,
    class_names=cfg.data.get("classes"), topk=5,
)
results = predictor.predict("cat.jpg")
```

要点：

- **请通过具体子类调用**。`BasePredictor` 自身的 `_postprocess()` 是未实现的抽象入口，
  直接 `BasePredictor.from_checkpoint(...)` 只能完成加载，`predict()` 会抛
  `NotImplementedError`。检测/分割分别用 `DetectionPredictor` /
  `SegmentationPredictor`，参数同上，另加各自的阈值/`return_probs` 参数。
- `image_size`、模型结构与 `num_classes` 都必须与训练一致；`from_checkpoint` 不会从
  checkpoint 推断这些信息，`strict=True` 也意味着缺少/多余的键都会直接报错，而不是静默
  加载部分权重。
- `predict()` 内部临时 `model.eval()` + `torch.no_grad()`，并在 `finally` 中恢复模型原来的
  `training` 状态；前向抛异常也会恢复，因此不会破坏后续继续训练。

### 3.3 `PreprocessPipeline.from_task` 单独使用

不构建模型、不加载权重，单独检查或复用推理预处理：

```python
import numpy as np
from dvisionix.inference import PreprocessPipeline

prep = PreprocessPipeline.from_task("detection", image_size=640)
print(prep.spec.to_dict())
# {'task_type': 'detection', 'image_size': 640, 'mean': [0.485, ...],
#  'std': [0.229, ...], 'scale': 0.00392156862745098}

tensor, meta = prep("a.jpg")
print(tuple(tensor.shape), tensor.dtype, tensor.device)
# (1, 3, 640, 640) torch.float32 cpu
print(meta.original_size, meta.processed_size, meta.scale)
# 例如原图 480×640 → (480, 640) (640, 640) (1.3333, 1.0)

# 也可以传 ndarray / PIL（ndarray 按 RGB 解释）
tensor, meta = prep(np.zeros((48, 64, 3), dtype=np.uint8))
```

`PreprocessPipeline.from_task` 支持的任务别名：`classification` /
`multi_task_classification` / `multi_label` / `linear_eval` 走分类预设；
`detection` 走检测预设；`segmentation` / `MaskFormer` / `mask_former` 走分割预设；
`simclr` 走 SimCLR 预设（只取一个视角）。**注意：预处理支持这些别名，不代表
`build_predictor` 支持对应任务的推理**（见 §7）。

如果已有自己的流水线，用 `PreprocessSpec.from_pipeline` 提取规格后直接包装：

```python
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.inference import PreprocessPipeline, PreprocessSpec

pipeline = ClassificationTransforms(
    train=False, image_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)
spec = PreprocessSpec.from_pipeline(pipeline, task_type="classification")
prep = PreprocessPipeline(spec, pipeline)   # spec 只是记录值，实际变换来自 pipeline
```

`from_pipeline` 会取第一个带 `.size` 的变换的 `size[0]` 作为 `image_size`，并从
`ImageNormalize` 提取 `mean/std/scale`；缺失时退回 ImageNet 默认值。

---

## 4. `Results` 字段与 JSON

### 4.1 字段表

| 字段 | 类型 / 形状 | 说明 |
|---|---|---|
| `task_type` | `str` | 任务类型；预测器原样写入 |
| `orig_size` | `tuple[int, int]`，`(H, W)` | 原图尺寸；构造时校验必须为正 |
| `image_path` | `str \| None` | 输入来自磁盘路径时的路径，否则 `None` |
| `probs` | `np.ndarray (C,) float32 \| None` | 分类 softmax 概率，和为 1；其他任务为 `None` |
| `topk` | `list[tuple[int, float]] \| None` | 分类 top-k，按概率降序，`k=min(topk, C)` |
| `boxes` | `np.ndarray (N, 4) float32 \| None` | 检测框 xyxy，**原图坐标**，已裁剪到图内 |
| `scores` | `np.ndarray (N,) float32 \| None` | 检测置信度 |
| `labels` | `np.ndarray (N,) int64 \| None` | 检测类别 id |
| `mask` | `np.ndarray (H, W) int64 \| None` | 分割类别图，**原图尺寸** |
| `meta` | `dict` | 附加信息：预测器给了 `class_names` 时写入 `class_names`；分割 `return_probs=True` 时写入 `probs` |

检测没有命中时，`boxes` 是 `(0, 4)`、`scores` 是 `(0,)`、`labels` 是 `(0,)`，不是
`None`；分割 `return_probs=True` 时 `meta["probs"]` 是原图尺寸的
`(C, H, W) float32` 概率，`mask` 仍是权威的类别图。

### 4.2 序列化

```python
results.to_dict()                 # numpy → list，不含原始图像
results.to_json()                # 返回 JSON 文本（indent=2, ensure_ascii=False）
results.to_json("a.json")        # 同时写文件，父目录不存在会自动创建
results.save("a.json")           # 等价于 to_json(path)
loaded = Results.load("a.json")  # 还原
```

`to_dict()` 不包含原始图像；分割的 `mask` 会转成嵌套 list，1024×1024 的 JSON 很大。
JSON 不保留 numpy dtype，`load()` 后 `probs` / `boxes` / `scores` 为 `float64`，
`labels` / `mask` 仍为 `int64`；数值本身不丢失。

### 4.3 JSON 示例

以下是一次 toy 模型（`SimpleCNN`，4 类）+ 随机输入的实际输出，**只用来展示字段结构**，
数值没有业务含义：

```json
{
  "task_type": "classification",
  "orig_size": [48, 64],
  "image_path": null,
  "probs": [0.24992378056049347, 0.2578132152557373, 0.25502151250839233, 0.23724153637886047],
  "topk": [[1, 0.2578132152557373], [2, 0.25502151250839233], [0, 0.24992378056049347], [3, 0.23724153637886047]],
  "boxes": null,
  "scores": null,
  "labels": null,
  "mask": null,
  "meta": {"class_names": ["cat", "dog", "bird", "fish"]}
}
```

检测和分割共用同一套字段：检测只填 `boxes` / `scores` / `labels`，分割只填 `mask`
（以及可选的 `meta["probs"]`）；未使用的字段为 `null`，字段名始终存在。

---

## 5. 反变换：回到原图坐标 / 尺寸

### 5.1 检测

`PreprocessMeta.scale = (sh, sw)`，含义是：

```text
sh = processed_height / original_height
sw = processed_width  / original_width
```

因此从「处理后坐标」回到原图的公式是**除以**比例，而不是乘：

```text
x_orig = clip(x_processed / sw, 0, W - 1)
y_orig = clip(y_processed / sh, 0, H - 1)
```

`DetectionPredictor` 对 `[x1, y1, x2, y2]` 分别按 `[sw, sh, sw, sh]` 做除法，再把 x 裁剪到
`[0, W-1]`、y 裁剪到 `[0, H-1]`。裁剪是必要的：`decode()` 在处理分辨率里允许取到
`img_w` / `img_h`，除完可能正好等于原图 `W` / `H`，而像素坐标最大值是 `W-1` / `H-1`。

检测/分割预设 `BoxSyncResize((image_size, image_size))` **直接拉伸到正方形**，不做
letterbox / padding，所以非正方形输入通常 `sh != sw`，必须按轴分别除；自定义 pipeline
若引入 padding，需保证它写回的 `meta.scale` 与坐标约定一致。`meta.scale` 缺失或非法时，
预测器退回用 `processed_hw / orig_size` 计算（预处理层也有同样兜底），避免除以 0。

### 5.2 分割

```text
logits (B, C, h, w)
  → argmax(dim=1)                         # (B, 1, h, w)，处理分辨率
  → F.interpolate(size=(H, W), mode="nearest")
  → (H, W) int64                          # 原图尺寸
```

注意顺序是**先 argmax 再最近邻上采样**，不是先插值 logits 再 argmax；最近邻只复制类别
索引，不会产生类别之间的混合值。

`return_probs=True` 时额外计算 `softmax` 并做 **bilinear** 插值到原图，写入
`Results.meta["probs"]`（`(C, H, W) float32`）。这是独立的概率图：它的 `argmax(0)` 在
边界附近可能与 `mask` 不完全一致（`mask` 用的是先 argmax 再 nearest），需要类别图时
请以 `mask` 为准。

### 5.3 分类

分类没有几何反变换：`softmax(logits[0])` 得到 `(C,)` 概率，`torch.topk` 取
`k=min(topk, C)` 个结果，按概率降序写入 `topk`（元素为 `(class_id, probability)`）。
`probs` 与 `topk` 的概率来自同一次 softmax，`probs.sum() == 1`。

---

## 6. 可视化

```python
vis = results.visualize(image=None, class_names=None)
```

返回 `(H, W, 3)` 的 **BGR uint8** 图像，可以直接 `cv2.imwrite` 保存：

- `image=None` 且 `results.image_path` 存在时读原图；否则在深色底（BGR 32）上画，
  画布尺寸取 `results.orig_size`；底图与 `orig_size` 不一致时先 resize；
- `image` 支持 `str` / `Path` / BGR ndarray / `PIL.Image`；ndarray 按 **BGR** 解释
  （与 `PreprocessPipeline` 的 RGB 约定相反，见 §1.3）；返回新图像，不修改传入底图。

各任务的绘制内容：

| 任务 | 绘制 |
|---|---|
| 分类 | 左上角逐行写 top-k `名称: 概率`（3 位小数）；没有 `topk` 时从 `probs` 取前 5 |
| 检测 | 每个框 + `label score`（2 位小数）；颜色按类别 id 固定 |
| 分割 | 按类别调色板叠色，透明度 0.5 |

`class_names` 为 `None` 或越界时显示类别 id。预测器给的 `class_names` 只会写进
`Results.meta["class_names"]`，`visualize` **不会自动读取它**；需要名称时显式传入
`results.visualize(class_names=results.meta.get("class_names"))`。CLI 的 `--save-vis`
会把配置里的 `data.classes` 直接传给 `visualize`，所以命令行产物默认带类别名。

---

## 7. 已知限制

如实列出，选择方案前请先确认不踩：

1. **`build_predictor` 只支持三个单任务**：`classification` / `detection` /
   `segmentation`。**不支持 `multi_task_classification`**：多任务模型输出的是
   `{任务: logits}`，没有「单一 top-k」语义，无法套用统一的 `Results`；也不支持
   `simclr`（无推理语义）。传这些类型会抛 `ConfigError`。预处理层支持这些别名，
   但那只是「按训练预设做确定性变换」，不等于多任务推理链路已完成。
2. **一次一张图**。`predict_many()` 内部是逐个 `predict()`，不做跨图堆叠或跨图批处理；
   不同原图尺寸的样本也无法直接拼成一个张量。批量吞吐优化不在 P4 范围内。
3. **未做 ONNX / TensorRT 推理与吞吐优化**。P4 只有 PyTorch 前向；`ONNXExporter`
   （见 [模型导出](model_export.md)）导出的是 `forward` 的原始输出，检测的 decode/NMS
   仍需要在推理端自行处理。ONNX / TensorRT 运行时推理与批量吞吐优化均**尚未实现**：
   P5 交付的是 ONNX **导出**（`tools/export.py`），不是 ONNX 推理。
4. **`PreprocessSpec` 从预设 pipeline 反推**。`from_pipeline` 只能提取 `ImageNormalize`
   的 `mean/std/scale`；自定义 mean/std 需自建 pipeline 并传给
   `PreprocessPipeline(spec, pipeline)`。**只改 spec 不会改变实际变换**：不传 `pipeline`
   时仍按 `spec.task_type` 构建预设流水线，spec 里的 mean/std 只是记录值。
5. **`from_checkpoint` 不推断模型结构**。模型必须先按训练配置构建好，`image_size` 也要
   显式传入且与训练一致；`strict=True` 会拒绝键缺失/多出的 checkpoint。
6. **CLI 输出按 stem 平铺**。递归目录中不同子目录的同名图片会写到同一个
   `<stem>.json` / `<stem>_vis.jpg`（后者覆盖前者）；请保证文件名唯一或分批处理。
7. **分割 JSON 未做压缩**。`mask` 会以嵌套 list 写入 JSON，高分辨率分割结果文件会很大；
   需要紧凑存储请自行保存 `.npy` 或额外编码。
8. **`--trusted` 是安全开关，不是兼容性开关**。它只解决「完整 Trainer checkpoint 含非张量
   状态」的问题；对来源不可信的 `.pt` 不应打开。文件损坏时即使 `--trusted` 也会失败。

---

## 8. 测试与验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python

# 推理专项
$PY -m pytest tests/test_inference -q

# 全量（P4 完成后实测 1066 passed / 1 skipped / 0 failed）
$PY -m pytest tests -q
```

`tests/test_inference/` 覆盖：

- `test_preprocess.py` / `test_results.py`：spec 提取、三类输入、meta、确定性、
  `to_dict` / `to_json` / `load` 往返与三种任务的可视化输出；
- `test_predictors.py`：toy 三模型端到端、框在原图内、mask 形状等于原图、概率和为 1、
  `from_checkpoint` 往返/损坏 checkpoint、预测后训练状态恢复；
- `test_infer_cli.py`：目录图片 + toy checkpoint + 最小配置，验证 JSON 与可视化落盘。

**真实数据端到端验收**（1 epoch checkpoint，仅用于确认链路跑通，**不作为模型效果结论**）：

- 分类：原图 870×870，5 类概率和为 1.0，top-1 类别为 2、概率 0.903；
- 检测：原图 439×439，检出 4 个框，全部落在原图范围内；
- 分割：原图 1024×1024，mask 为 `(1024, 1024) int64`，取值样例 `[0, 3, 5, 10]`。

反例：对完整 Trainer checkpoint 不加 `--trusted` 会抛 `CheckpointError`，并在提示中明确
要求 `trusted=True`（见 §2.3）。

---

## 9. 相关文档

- [快速开始](quick_start.md)：安装、训练与最短推理命令；
- [训练契约与自适应训练](training_contracts.md)：任务/模型/数据契约；
- [目标检测](detection.md)：检测器与 `decode()` 契约；
- [语义分割](segmentation.md)：分割数据格式与训练；
- [模型导出](model_export.md)：ONNX 导出（不含推理运行时）；
- [P4 推理链路契约](design/v2_p4_contract.md)：字段与行为的实现契约；
- [CHANGELOG](../CHANGELOG.md)：P4 变更记录。
