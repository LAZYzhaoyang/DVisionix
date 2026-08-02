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

- 无阻塞性待办；后续新功能一律进入「未来规划」排期。

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

## 七、版本记录（精简）

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
