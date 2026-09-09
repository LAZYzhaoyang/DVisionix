# DVisionix 开发计划（CodePlan）

> 本文档是算法库的**唯一规划依据**：描述项目状态、已实现功能、当前计划完成进度与未来发展规划。
> 开发必须遵守「model 模块调用规则（R1-R7）」；使用文档见 `docs/`，安装与快速开始见 `README.md`。

---

## 一、项目定位与目标

DVisionix 是一个**个人使用的 PyTorch 视觉算法工具箱**：

1. **配置驱动**：所有可替换组件（model / data / task / loss / metric / transform）均由 YAML 配置构建。
2. **组件注册**：统一 Registry 机制，新增组件「注册即可用」，不改 `__init__` 硬编码。
3. **模型组件化**：`backbones → necks → heads → detectors` 三段式，后处理与模型解耦，自由组合。
4. **训练引擎现代化**：AMP、梯度累积、DDP、完整 resume、EMA、蒸馏、超参搜索、实验管理。
5. **正确性优先**：检测 mAP / NMS 对接成熟实现；指标与评估结果可复现。
6. **代码卫生**：统一编码（UTF-8 无 BOM）、统一命名、全库注释（作者 + 用途 + 输入输出说明）。

---

## 二、项目状态总览（v1.0.0 基线）

- **版本**：v1.0.0（功能基线，API 冻结进入稳定期）。
- **测试**：286 passed + 2 skipped（多卡冒烟需 2+ GPU 自动跳过）；ruff / black 全绿。
- **工作目录**：默认 `~/dvisionix_runs/<experiment>/<时间戳>-<配置哈希>/`（代码库外隔离）。
- **入口**：`tools/train.py`（训练）/ `tools/hparam_search.py`（超参搜索）/ `ONNXExporter`（导出）。

---

## 三、已实现功能矩阵

### 1. 数据模块（data）
- `Sample` 协议 + `BaseDataset` 统一驱动（image / boxes / mask 自动标准化）。
- 原子变换：图像（Resize/Crop/Flip/ColorJitter/Normalize/ToTensor）、几何同步（BoxSync*）、标签、第三方适配（albumentations）。
- 公开数据集工具箱：CIFAR / ImageNet / COCO / VOC / Cityscapes / ADE20K / ImageFolder。
- 自定义数据集模板：`data/datasets/custom.py`；合成数据快速验证（tools/train.py 内置）。

### 2. 模型模块（models）
**骨干（12）**：Sequential / ConvNeXt / ConvNeXtV2 / CSPDarknet / MobileNetV3 / EfficientNetLite /
ViT / Swin / SwinV2 / MiT / TimmBackbone / TimmClassifier。

**颈部（3）**：FPN / PANet / PixelDecoder。

**分类头（11）**：Cls / ArcFace / CosFace / SphereFace / AdaFace / NormFace / CurricularFace /
PartialFC / CircleLoss / MultiLabel / SimCLR。

**分割头（14）**：Seg / FCN / DeepLabV3 / DeepLabV3Plus / UNet / SegFormer / SegFormerV2 / SegFormerV3 /
PSP / UPerNet / BiSeNet / MaskFormer / Mask2Former / SwinUNetDecoder。

**检测头（10）**：Det 系（FCOS / RetinaNet / YOLO / DETR / RTDETR / RTDETRFull / DeformableDETR /
CenterNet / NMSFreeYOLO / DINO）。

**检测器（12）**：SingleStageDetector 脚手架 + FCOS / RetinaNet / YOLOv8 / YOLOv9(PGI) / YOLOv10(NMS-free) /
DETR / RT-DETR / RT-DETR-full / DeformableDETR / CenterNet / DINO-lite；decode 与模型同文件。

**损失（25+）**：分类（CE / Focal / BCE / Circle / InfoNCE / 蒸馏 / 特征蒸馏）+ 分割（Dice / CE+Dice /
MaskFormer）+ 检测（Objectness / Grid / SigmoidFocal / FCOS / RetinaNet / YOLO / one-to-one / CenterNet /
YOLOv9 / DETR / DINO-LFT）+ assigner（Grid / FCOS / MaxIoU / ATSS / TaskAligned）+ matcher（匈牙利）。

**组合器**：LinearClassifier（classifiers/）、SegmentationModel / SwinUNet（segmenters/）。

**教学模型（toy）**：SimpleCNN / SimpleSegmentationModel / GridDetectionModel / DetHead（与生产组件隔离）。

### 3. 训练工程（training）
- **Task 系统**：BaseTask + Classification / Detection / Segmentation / MultiLabel / SimCLR /
  LinearEval（冻结 backbone + L2 归一化线性头）/ MaskFormer（实例/全景）。
- **Trainer**：DDP、AMP（GradScaler）、梯度累积、完整 resume（model/optimizer/scheduler/scaler/rng/callbacks/task）、
  torch.compile（失败降级）、channels_last、best_metrics.csv 导出。
- **Callback**：ProgressBar / ModelCheckpoint（best/last/epoch 存档）/ EarlyStopping / EMA（decay warmup + 最终导出）/
  DistillCallback（logits + 特征提取）。
- **优化器/调度器**：adam/adamw/sgd/rmsprop；cosine/step/reduce_on_plateau/linear_warmup。
- **工程工具**：work_dir 隔离（配置哈希后缀）、`--resume auto`、hparam_search、`export_best_onnx`。

### 4. 指标与评估（metrics）
- 分类：Accuracy / TopK / Precision / Recall / F1（macro/micro/weighted/none）。
- 分割：mIoU / Pixel Accuracy / Dice 等（ignore_index / per_class）。
- 检测：COCO 风格 mAP@0.5 / mAP@0.5:0.95（内置实现，torchmetrics 可选）。
- 实例/全景：Mask mAP、PanopticQuality（PQ / SQ / RQ）。
- MetricCollection 组合 + `get_preset_metrics` 预设。

### 5. 工具与导出（tools / export / utils）
- `tools/train.py`：统一训练入口（任务自动映射、合成数据、resume、DDP、ONNX 导出）。
- `tools/hparam_search.py`：参数网格/随机采样，逐 trial 独立进程，汇总 search_results.csv。
- `ONNXExporter`：trace/dynamo 后端、单/多输入、dict 多输出、动态 batch、精度验证、归一化元数据。
- 日志：console + file + JSONL + TensorBoard（TrainingLogger）。

---

## 四、model 模块调用规则（R1-R7）— 强制开发约束

> 新增/修改模型代码时必须遵守；违反即视为缺陷。

### 分层调用图（依赖方向 = import 方向，自上而下）

```
组合层：detectors/ · classifiers/ · segmenters/        ← 最上层（聚合组件）
   │  import：layers · postprocess · backbones · necks · heads
   ▼
组件层：heads/ · losses/ · backbones/ · necks/
   │  import：layers · postprocess（heads 可经 Registry 注入 necks 组件）
   ▼
最底层：layers/ · postprocess.py                      ← 只依赖 torch/registry
```

### 各层职责表

| 层 | 模块 | 职责 | 允许 import | 禁止 import |
|---|---|---|---|---|
| 最底层 | layers/ | 通用算子/层（norm/attention/anchors/patch ops/CSP/ELAN/可变形…） | torch、registry | 任何上层模块 |
| 最底层 | postprocess.py | NMS/IoU 原语 + 共享契约解码器 | torch | 任何上层模块 |
| 组件层 | backbones/ | 12 种骨干 | layers；**同包共享基类 feature.py** | heads/detectors/losses/兄弟骨干 |
| 组件层 | necks/ | FPN/PANet/PixelDecoder | layers | heads/detectors/losses/兄弟 neck |
| 组件层 | heads/ | 各类头（每头一文件） | layers；可选 Registry 注入 necks 组件 | backbones/detectors/losses/兄弟 head |
| 组件层 | losses/ | 各任务损失 + assigner/matcher | layers、postprocess；**同包支撑模块（base/box_loss/matcher）** | backbones/necks/heads/detectors |
| 组合层 | detectors/、classifiers/、segmenters/ | 组装 backbone+neck+head 为可用模型 | 下层全部 | 无 |

### 规则条目

- **R1 依赖单向（自顶向下）**：只允许上层 import 下层；禁止下层 import 上层；依赖方向与调用图一致。
- **R2 同级隔离**：backbones / necks / heads / detectors 各自内部兄弟模块互不 import；共享实现一律下沉
  到 layers/、necks/ 或 postprocess.py。
  - **R2 例外（子包内支撑模块）**：同一子包内的**共享基类 / 支撑模块**允许被兄弟 import，包括：
    `backbones/feature.py`（FeatureBackboneBase）、`heads/detection/yolo.py`（NMSFreeYOLOHead 继承 YOLOHead）、
    `losses/base.py`、`losses/classification.py`（分割损失复用 CrossEntropy）、`losses/detection/matcher.py`
    （MaskFormer 复用 HungarianMatcher）、`losses/detection/box_loss.py` 等。新增此类复用须先在本文档登记。
- **R3 职责边界**：heads 不依赖 backbones / detectors / losses；heads 可使用 layers 组件，并经 Registry
  注入 necks 通用解码器组件（如 pixel_decoder）；anchors/bbox 编解码归属 layers/。
- **R4 decode 策略**：模型专属解码与其 head/detector 同文件（如 fcos_decode）；多模型共享、契约一致的
  解码纯函数放 postprocess.py（如 maskformer_decode）；每个模型保留 decode() 实例方法做薄桥接。
- **R5 组合器经 Registry 构建**：classifiers / segmenters / detectors 通过 BACKBONES/NECKS/HEADS 注册表
  构建下层组件，不直接 import 具体类。
- **R6 新增组件流程**：新算子 -> layers/ 或 necks/ -> 新 head/backbone 只引用下层 -> 每头一文件 -> 注册即用。
- **R7 组合器子包**：组合器（classifiers / segmenters / detectors）均为子包、每类一个文件；新增组合模型
  在对应子包新建文件并导出。教学模型归入 `models/toy/`，不与生产组件混放。

---

## 五、当前计划与完成进度

### ✅ 已完成

| 阶段 | 内容 | 版本 |
|---|---|---|
| 阶段 0 | 代码卫生：编码统一、BOM 清理、死代码删除、版本声明修正 | v1.0.0 |
| 阶段 1 | Registry + Config 配线、Config schema、CLI 覆盖 | v0.2.0 |
| 阶段 2 | 训练子系统重构：Task 组件化、loss 迁移到模型层、DDP/resume/work_dir | v0.3.0 |
| 阶段 3 | 模型模块丰富：骨干/颈部/头/检测器/损失体系 | v0.4.0-v0.9.0 |
| 阶段 4 | 组合器目录化 + model 分层重构 + 调用规则 R1-R7 入册 | v0.13.0 |
| 阶段 5 | 中期模型扩充：ConvNeXtV2/EfficientNetLite/MiT/SwinUNet/YOLOv11 | v0.14.0 |
| 阶段 6 | SwinV2/DeformableV2/SegFormerV3 批次 2 | v0.15.0 |
| 阶段 7 | DINO-lite + 线性评估 + 训练工程 P1 | v0.16.0 |
| 阶段 8 | 训练工程 P2（超参搜索/特征蒸馏）+ P3（性能开关/实验管理）+ DINO-LFT | v0.17.0 |
| 阶段 9 | **v1.0.0 收尾**：全库审查、注释、文档、README/CodePlan 规范化 | v1.0.0 |

### 🔄 进行中 / 待办

- 当前进入 v1.1 稳定性与工程优化阶段，优先执行第七章 P0/P1 项目。
- P0/P1 完成并通过验收前，暂停继续扩充模型家族，避免扩大未验证行为面。

---

## 六、未来发展规划（按优先级）

> 以下计划**默认推迟**，仅按明确指示实施（实施后同步更新本文件与文档）。

### P0 — 多卡实验验证（medium）
- **背景**：DDP 路径已实现（`Trainer(strategy="ddp")` + `torchrun` + DistributedSampler + all_gather 指标），
  但尚未在真实多卡环境验证。
- **实现步骤**：① 在 2+ GPU 机器上跑 `torchrun --nproc_per_node=2 tools/train.py --config ... --devices 0,1`；
  ② 验证各 rank 批数一致（drop_last）、指标 all_gather 无死锁、checkpoint 仅 rank0 保存；
  ③ 补充端到端 DDP 训练一致性测试（单卡 vs 多卡 loss 曲线对齐）。
- **验收**：`tests/test_training/test_ddp_smoke.py` 在多卡环境全绿。

### P1 — 指标 torchmetrics 迁移（low）
- **背景**：内置指标（mAP / mIoU / PQ）正确性已由测试保障，torchmetrics 作为可选后端。
- **实现步骤**：① `metrics/detection.py` 增加 `backend="torchmetrics"` 分支；② 分割/全景指标同法；
  ③ 保持 `BaseMetric` 接口不变（update/compute/reset），配置 `metrics.backend` 切换。
- **验收**：内置与 torchmetrics 后端在同一合成数据上结果一致（容差 1e-4）。

### P2 — 模型库继续扩充（medium）
- 骨干：更多 Transformer 变体（ConvNeXtV3 趋势、MoE 视觉骨干等，按最新论文跟进）。
- 检测：YOLO 系列完整化（YOLOv12 等）、DETR 家族（Co-DETR / DINOv2 集成）。
- 分割：Mask2Former 完整训练增强、SAM 类交互分割（另行立项）。
- 新增组件遵循 R1-R7，注册即用，配套合成数据配置与测试。

### P3 — 训练工程增强（low）
- 自动混合精度扩展（bf16）、DeepSpeed/FSDP 大模型训练（按需）。
- 实验管理 Web 化（可选：接入 wandb/MLflow 日志后端）。

---

## 七、v1.1 稳定性与工程优化计划

> 本节是 v1.0.0 基线之后的具体执行计划。优先修复会导致官方示例失败、训练结果错误、分布式评估错误和非 editable 安装失败的问题，再进行性能和发布能力建设。每项任务必须同时提交实现、回归测试和文档更新；未满足验收标准不得标记为完成。

### 7.1 当前审查结论

v1.0.0 的架构方向成立，但当前基线仍存在以下已确认或高风险问题：

- `dvisionix/data/datasets/custom.py:39-44` 将 `staticmethod` 对象写入实例属性，检测/分割 `CustomDataset` 的 `collate_fn` 可能不可调用。
- `dvisionix/config/schema.py:63` 未包含 `simclr`，但 `configs/classification/simclr_synthetic.yaml:4` 使用 `task_type: simclr`，官方 SimCLR 配置会在 schema 阶段失败。
- `dvisionix/registry.py:99` 将构造参数 `name` 与 Registry 注册名混用，`TimmBackbone/TimmClassifier` 的模型名称可能被静默丢弃。
- `dvisionix/training/trainer.py:54-63` 对 DDP 聚合后的嵌套 tuple 使用 `extend`，检测/实例分割的多字段预测结构会被破坏。
- `dvisionix/models/losses/detection/losses.py:237,460` 的 FCOS/YOLO 回归损失在 `use_giou=True` 时同时累加 GIoU 与 L1，配置语义和实际训练信号不明确，需确认并修正。
- `dvisionix/data/transforms/__init__.py:123-126` 检测管线先 resize 到目标尺寸再执行同尺寸 crop，随机裁剪实际不生效。
- `dvisionix/data/transforms/image.py:175-189` 可能将 mask 转为 float，导致分割标签无法满足 `CrossEntropyLoss` 的 long dtype 契约。
- `dvisionix/training/evaluation.py:94-100` 在首张图片无预测 mask 时把目标尺寸退化为 `(1, 1)`，mask AP 结果可能失真。
- `dvisionix/export/onnx_exporter.py:29-40,150-176,253-273` 对检测模型嵌套输出、CUDA verify 和 dynamo 导出参数的支持不完整。
- `dvisionix/models/heads/detection/dino.py:119-123` 硬编码 stride=4 推断图像尺寸，非 stride-4 backbone 的去噪目标可能错误。
- `dvisionix/training/trainer.py:434-445,450-480` 的 DDP 检测聚合和独立 `validate()` 路径尚未形成统一的全局指标计算契约。
- `setup.py:23-24` 未明确打包 `dvisionix/config/defaults/*.yaml`；非 editable wheel 安装后 `Config.from_default()` 存在失败风险。
- `setup.py`、`requirements.txt` 和 `pyproject.toml` 的依赖、版本和检查配置存在多源维护；CI 不运行 mypy、coverage 和依赖安全扫描。
- 当前环境执行 `python -m pytest tests -q` 时因未安装 pytest 无法复现 README 中的历史测试数字，因此 README/CodePlan 中的测试状态必须改为以 CI 运行结果为准。

### 7.2 P0：核心正确性修复

#### P0-1 修复 CustomDataset 的 collate 函数

**涉及文件**：`dvisionix/data/datasets/custom.py`、`tests/test_data/test_data.py`。

**实施步骤**：

1. 将 detection/segmentation 分支分别改为直接保存 `detection_collate` 和 `segmentation_collate` 函数。
2. 保留用户显式传入 `collate_fn` 的最高优先级。
3. 构造后断言 `callable(ds.collate_fn)`，禁止静默返回不可调用对象。
4. 使用 `DataLoader(..., collate_fn=ds.collate_fn)` 完成检测和分割 batch 冒烟测试。

**验收标准**：三种任务的 `CustomDataset` 均可构造；检测/分割 DataLoader 能取出一个 batch；boxes、labels、mask 的结构和 dtype 正确。

#### P0-2 修复 SimCLR 任务类型配置

**涉及文件**：`dvisionix/config/schema.py`、`tools/train.py`、`tests/test_config/test_schema.py`、SimCLR 冒烟测试。

**实施步骤**：

1. 将任务类型常量集中到单一模块，schema 和训练入口共享同一来源。
2. 将 `simclr` 加入合法任务类型。
3. 为 SimCLR 增加 temperature、batch size 等关键参数校验。
4. 用官方合成配置完成配置加载、数据构建、前向、反向和 checkpoint 保存。

**验收标准**：`python tools/train.py --config configs/classification/simclr_synthetic.yaml --work-dir <tmp>` 能完成一个 epoch。

#### P0-3 修复 Registry 的 `name` 参数冲突

**涉及文件**：`dvisionix/registry.py`、timm backbone、Registry 测试。

**实施步骤**：

1. `type` 始终表示注册表键。
2. 仅在没有 `type` 时使用 `_name_` 作为注册表键。
3. 普通 `name` 原样传入构造函数。
4. 对旧配置中使用 `name` 作为注册键的情况给出迁移错误，不再静默猜测。
5. 测试 `type`、`_name_`、构造参数 `name` 和错误配置。

**验收标准**：`BACKBONES.build({"type": "timm_backbone", "name": "resnet18"})` 必须实际构造 resnet18。

#### P0-4 修复 DDP 嵌套结果聚合

**涉及文件**：`dvisionix/training/trainer.py`、DDP 指标测试。

**实施步骤**：

1. 将 `_concat_objects` 改为递归合并：Tensor 使用 `torch.cat`，list/tuple 按位置递归，dict 按 key 递归。
2. 对 rank 间结构、tuple 长度和 dict key 不一致显式报错。
3. DDP 训练验证和 `Trainer.validate()` 复用同一 gather/reduce helper。
4. 增加分类 Tensor、检测 `(boxes,scores,labels)` 和分割嵌套结构测试。
5. 使用 CPU `gloo` 双进程作为 CI 回归路径，再在 GPU 环境验证 NCCL。

**验收标准**：单进程与双进程全局指标在 `1e-6` 内一致；无死锁、解包错误或只统计 rank0 分片。

#### P0-5 修复损失函数语义和归一化

**涉及文件**：`dvisionix/models/losses/detection/losses.py`、检测 loss 测试。

**实施步骤**：

1. 明确 `use_giou=True` 是替代 L1 还是组合 GIoU+L1。
2. 推荐改为显式 `giou_weight`、`l1_weight`，旧参数提供迁移逻辑。
3. 统一 FCOS、YOLO、RetinaNet 的正样本归一化。
4. 为无正样本 batch 定义无 NaN 的明确行为。
5. 使用固定预测和目标手算期望，确认每个分量只计算一次。

**验收标准**：测试可区分 GIoU、L1 和组合模式；空目标不产生 NaN；loss 下降测试继续通过。

### 7.3 P1：训练、评估和导出可靠性

#### P1-1 修复检测增强和裁剪尺寸契约

**涉及文件**：transforms preset、image/geometric transforms 及其测试。

**实施步骤**：

1. 检测训练 pipeline 改为先放大再随机 crop，或删除无效 crop。
2. `RandomCrop/CenterCrop` 增加 `on_small=error|pad|resize`，禁止静默返回错误尺寸。
3. 几何变换校验 boxes 和 labels 数量一致。
4. 增加图像、boxes、mask 同步变换的固定随机种子测试。

**验收标准**：裁剪有非零偏移；输出尺寸一致；boxes 不越界；labels 与 boxes 数量一致。

#### P1-2 修复 mask dtype 和 mask AP

**涉及文件**：`image.py`、`labels.py`、`training/evaluation.py` 及测试。

**实施步骤**：

1. `ToTensor` 对 image 固定输出 float32，对 mask 固定输出 long。
2. `MaskToTensor` 无论输入是否为 Tensor都校验并转换为 long。
3. mask 加载校验维度、类别值和 ignore_index。
4. `evaluate_mask_ap` 从原始 `pred_masks` 或真实 image size 获取目标尺寸。
5. 覆盖首图、部分图片、全部图片零预测场景。

**验收标准**：mask dtype 为 long；空预测不把 target 缩放到 1x1；指标结果稳定可解释。

#### P1-3 修复 DINO 尺寸来源

**涉及文件**：DINO head、detector/task 装配和 DINO 测试。

**实施步骤**：

1. 从 batch image 读取真实 `(H,W)`，显式传入 head。
2. feature map 推断仅作兼容 fallback，且要求 stride 元信息。
3. 覆盖 stride=2、stride=4 和非方形输入。

**验收标准**：去除硬编码 `*4` 主路径，不同 stride 下 denoising target 坐标正确。

#### P1-4 完善 ONNX 导出契约

**涉及文件**：`dvisionix/export/onnx_exporter.py`、导出测试。

**实施步骤**：

1. 增加递归输出 flatten，支持 dict/list/tuple 嵌套，并保存输出路径映射。
2. verify 统一使用 `detach().cpu().numpy()`。
3. 保存并恢复模型 device 和 train/eval 状态。
4. 明确 trace/dynamo 对 names 和 dynamic axes 的支持范围，不支持时显式报错。
5. 为分类、分割和一个多尺度检测模型增加 ONNX Runtime 验证。

**验收标准**：三类任务均可导出；动态 batch 可实际执行；导出器不改变原模型状态。

#### P1-5 统一 Trainer 验证和 checkpoint 状态

**涉及文件**：Trainer、EMA、checkpoint、resume 测试。

**实施步骤**：

1. 训练中验证和独立 `validate()` 复用同一 evaluator。
2. checkpoint 增加 schema version、task/model type 和配置 hash。
3. 支持 `ema_best.pt` 或保证 best checkpoint 包含对应 EMA 状态。
4. EarlyStopping 的 best value、等待计数和恢复状态全部持久化。
5. 默认拒绝配置不匹配的 resume，允许显式 override。
6. 修正尾部不足一个 accumulation window 时的 loss 分母。

**验收标准**：resume 前后 optimizer/scheduler/RNG/EMA/early stopping 一致；独立 validate 与训练时指标一致。

### 7.4 P1：测试、CI 和发布基础设施

#### P1-6 建立真实测试门禁

**涉及文件**：`.github/workflows/ci.yml`、`pyproject.toml`、`tests/conftest.py` 和现有测试目录。

**实施步骤**：

1. CI 覆盖 Python 3.10、3.11、3.12。
2. 增加 mypy，并将配置版本修正为项目最低 Python 版本。
3. 增加 coverage，初始门槛 60%，稳定后提升至 75% 以上。
4. 启用 strict markers/config，注册 unit/integration/slow/cuda/ddp marker。
5. 按功能域整理 `test_v*.py`，不再用历史版本号表达覆盖范围。
6. 增加四类任务 CLI 端到端测试和 CPU DDP 测试。

**CI 顺序**：锁定依赖安装 -> ruff -> black -> mypy -> unit -> integration -> coverage -> wheel 安装测试 -> pip-audit。

**验收标准**：PR 可直接看到测试、覆盖率、类型、打包和安全结果；测试状态以 CI 为准。

#### P1-7 修复依赖和打包一致性

**涉及文件**：`pyproject.toml`、`setup.py`、`requirements.txt`、package data、安装测试。

**实施步骤**：

1. 迁移到 `pyproject.toml [project]`，消除打包元数据双源。
2. 依赖分为 core/dev/export/datasets extras。
3. 使用 uv 或 pip-tools 生成 lockfile，CI 使用锁定版本。
4. 明确 torch/torchvision 的兼容矩阵和 CPU/CUDA 安装策略。
5. 将 `dvisionix/config/defaults/*.yaml` 纳入 wheel/sdist。
6. 统一版本号来源，补充 LICENSE 和 CHANGELOG。
7. 增加 wheel 干净环境安装后调用 `Config.from_default()` 的测试。

**验收标准**：构建 wheel 后在新虚拟环境安装，默认配置加载和最小 import 均成功。

#### P1-8 增加安全和发布流程

**涉及文件**：CI、Dependabot、release workflow、checkpoint loader 和安全文档。

**实施步骤**：

1. CI 执行 pip-audit，增加 Dependabot。
2. Actions 固定 SHA 或至少配置最小权限、超时和并发取消。
3. 建立 tag -> build -> smoke test -> TestPyPI/PyPI 发布流程。
4. 区分纯权重与完整 checkpoint：默认 `weights_only=True`；完整 pickle 状态需显式声明可信。
5. checkpoint 加载前校验结构、版本和 tensor 类型。

**验收标准**：发布 wheel 可在干净环境运行；不可信 checkpoint 风险和 API 策略有明确文档。

### 7.5 P2：性能和长期维护

#### P2-1 优化 EMA 和数据搬运

1. EMA shadow tensor 原地更新，避免每 batch 新建 tensor。
2. 区分浮点参数和非浮点 buffer。
3. DataLoader 暴露 pin_memory、persistent_workers、prefetch_factor。
4. 支持 non_blocking 设备搬运。
5. 用 profiler 对 EMA、加载和拷贝建立基线。

**验收标准**：EMA 开销低于训练总耗时 3%，或提供无法达到时的基准说明。

#### P2-2 优化 DDP 评估通信

1. 保留 `all_gather_object` 作为 fallback。
2. Tensor 结果使用 padding + valid count + `dist.all_gather`。
3. 大型检测评估支持 rank 分片落盘、rank0 汇总。
4. 记录通信耗时和样本数量。

**验收标准**：结果与单卡一致；无空预测/变长结果死锁；通信耗时有明确下降。

#### P2-3 优化 mAP、PQ、matcher 和 assigner

1. mAP 复用排序和 IoU 中间结果，减少阈值重复计算。
2. PQ 使用类别过滤、bbox 粗筛和分块 overlap，禁止超大 `(P,G,H,W)` 中间张量。
3. matcher 减少 GPU->CPU 同步，必要时先做候选裁剪。
4. 向量化 TaskAlignedAssigner 和 OneToOneYOLOLoss 的 Python 循环。
5. 用固定规模输入记录耗时、峰值内存和正确性。

**验收标准**：优化前后指标在容差内一致；高分辨率全景评估无异常内存峰值。

#### P2-4 完善运行时契约和 API

1. 实现 `Sample._KNOWN_KEYS` 检查，或删除未实现的文档承诺。
2. 明确 RGB/BGR，令 `ImageMode` 参与数据加载校验。
3. 让 `provides_normalization` 真正阻止重复归一化。
4. `BaseModel.get_device()` 处理无参数模型。
5. `assert` 改显式异常，out_indices 越界禁止静默取模。
6. MetricCollection 对重复名称报错。
7. 库代码中的 `print()` 统一为 logger。
8. 将 TensorBoard 标量真正接入 Trainer 生命周期。
9. 减轻顶层 import，避免只使用 config 时无条件加载完整视觉栈。

**验收标准**：错误输入在构建阶段失败；文档契约都有运行时代码或测试证明。

### 7.6 执行顺序与完成定义

**第一周：恢复正确性**

- 完成 P0-1 至 P0-5。
- 跑通分类、检测、分割、SimCLR 合成配置。
- 每个缺陷都有回归测试。

**第二周：训练与评估闭环**

- 完成 P1-1 至 P1-5。
- 建立 CPU DDP、resume、EMA、mask AP、ONNX 检测输出测试。

**第三周：CI、依赖与打包**

- 完成 P1-6 至 P1-8。
- 建立 coverage、mypy、pip-audit 和 wheel 干净环境安装门禁。

**第四周及以后：性能优化**

- 完成 P2 项目；先建立 benchmark，再修改实现。
- P0/P1 完成前暂停继续扩充模型家族，避免扩大未验证行为面。

#### Definition of Done

任务只有同时满足以下条件才可标记为完成：

1. 实现已提交，没有通过静默 fallback 掩盖错误。
2. 至少有一个针对原缺陷的回归测试。
3. 相关单元和集成测试通过。
4. 文档、配置示例和 API 行为一致。
5. 性能改动有固定输入的前后 benchmark。
6. 分布式改动有单卡/多卡一致性证据。
7. 发布改动有干净环境 wheel 安装证据。

---

## 八、版本记录（精简）

| 版本 | 里程碑 |
|---|---|
| v1.0.0 | 功能基线：全库审查/注释/文档规范化，API 冻结 |
| v0.17.0 | 训练工程 P2+P3、DINO look-forward-twice |
| v0.16.0 | DINO-lite、线性评估、训练工程 P1 |
| v0.15.0 | 组合器目录化、SwinV2/DeformableV2/SegFormerV3 |
| v0.14.0 | 中期模型扩充（骨干/分割/检测批次 1） |
| v0.13.0 | model 分层重构 + layers 统一 + 调用规则 R1-R6 |
| v0.12.0 | ViT/Swin 骨干、YOLOv9-lite(PGI) |
| v0.11.0 | 内置骨干体系、SimCLR、分割增强 |
| v0.10.0 | YOLOv7/v10、CenterNet、BiSeNet、Circle/SimCLR 分类头 |
| v0.9.0 | 全景评估、RT-DETR-full、Mask2Former 完整版 |
| v0.8.0 | PSP/UPerNet/DeepLabV3+、DeformableDETR、度量学习头 |
| v0.7.x | decode 归位、组合性验证 |
| v0.6.0 | Mask2Former 完整版、EMA/蒸馏回调、CI |
| v0.5.0 | YOLO/DETR、SegFormer/MaskFormer、度量学习 |
| v0.4.0 | 模型模块丰富（FCOS/RetinaNet/UNet/DeepLab 等） |
| v0.3.0 | 训练子系统重构（Task/Trainer/DDP/resume） |
| v0.2.0 | 组件化重构（Registry + 配置驱动入口） |
