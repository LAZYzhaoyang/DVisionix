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

- **版本**：v1.1 开发中（v1.0.0 功能已冻结；稳定性与工程修复见第七章）。
- **测试**：v1.0.0 基线为 `286 passed + 2 skipped`、`ruff` / `black` 全绿 —— 已实测复现。
  完整基准数字见第五章「v1.0.0 基线实测数据」。**测试状态一律以 CI 结果为准**，本文件不手写数字。
- **已核实的关键缺口**：19 个官方配置中 3 个无法训练、`decode()` 统一契约实际 2/11 违约、
  EMA 无配置入口、docstring 3 处损坏等，共 20 项，完整清单与证据见 **7.1.2**；
  v1 版计划中 4 条不成立的论断见 **7.1.1**。
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
- **Callback**：ProgressBar（Trainer 默认注入）/ ModelCheckpoint（best/last/epoch 存档）/
  EarlyStopping / EMA（decay warmup + 最终导出，**可经 `training.ema` 配置开启**）/
  DistillCallback（logits + 特征提取，**仅编程式 API** —— 需要教师权重实例，配置入口不支持）。
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

### ✅ 已完成（v1.0.0 及以前）

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

### 📊 实测数据（唯一可比基准）

> 于 `conda env dvisionix`（Python 3.14.6 / torch 2.12.0 / CPU-only）实测，
> 作为全部改动的对照基准。**任何任务的完成判定都必须与这组数字对比。**

| 指标 | v1.0.0 基线（HEAD `e20d5e9`） | 当前（阶段 0-2 完成后） |
|---|---|---|
| 测试 | `286 passed, 2 skipped`（69.4s），collected = 288 | `403 passed, 2 skipped`（186.2s），collected = 405 |
| 测试构成 | 纯组件级 | 组件级 + 配置 E2E 门禁 17 + 契约/回归/CPU-DDP 共 100 |
| 静态检查 | `ruff` / `black --check` 全绿 | `ruff` / `black --check` 全绿 |
| 官方配置 E2E（19 个各跑 1 epoch） | **14 PASS / 3 FAIL / 2 模板不可跑** | **17 PASS / 0 FAIL / 2 模板不可跑** |
| 跑不通的配置 | `classification/simclr_synthetic`、`detection/centernet_synthetic`、`detection/yolov10_synthetic` | 无 |
| 不可跑的模板（非缺陷） | `classification/hparam_search`（用 `tools/hparam_search.py`）、`classification/linear_eval`（占位 checkpoint 路径） | 同左 |
| CPU 双进程 DDP 一致性 | 无此测试（`test_ddp_smoke` 需 2+ GPU，恒跳过） | ✅ 单进程 vs 2 进程全局指标在 `1e-6` 内一致 |

> 说明：测试数从 288 增至 405 全部是**新增门禁与回归测试**，不是实现膨胀；
> 其中 17 条为配置端到端门禁（`-m "not slow"` 可跳过）、2 条为 CPU gloo 双进程、
> 其余为契约与回归测试。无既有测试被删除或替换。

### 🔄 当前状态：v1.1 稳定性与工程优化

- **第七章是本项目唯一执行计划。**
- **阶段 0（门禁）、阶段 1（修复跑不通的官方配置）、阶段 2（P0 正确性）已完成**，
  见 7.2 / 7.3 / 7.4 的完成标记。官方配置由 14/17 可跑提升到 17/17 全绿；
  CPU 双进程 DDP 一致性已建立。
- **阶段 3-5 未开始**。阶段 3 完成前仍暂停扩充模型家族。
- **v1.1 早期一批未提交改动已整体回退**（回退原因与更正见 7.1.1，
  补丁留档于 `.dev_archive/wip-v1.1-partial.patch`，已在 `.gitignore` 中排除）。
- 阶段 2 已重新实现该批次中**诊断正确且实现无误**的两项（DDP 递归聚合、FCOS/YOLO 重复累加 L1），
  并补齐其缺失的归一化与测试。

### ⏸️ 环境阻塞

- 多卡 NCCL 验证（原第六章 P0）：当前环境 `torch.cuda.is_available() == False`，无法执行；
  替代方案（CPU gloo 双进程）已落地，解除条件见 **7.8**。

### ⚠️ Windows 运行须知（影响 CPU 分布式测试）

在 Windows 上直接用解释器绝对路径运行（**未激活 conda 环境**）时，
`Library\bin` 不在 `PATH`，子进程 `import torch` 会随机报
`ImportError: DLL load failed while importing _C`。
请先激活环境（`conda activate dvisionix`）再运行测试；
`tests/test_training/test_ddp_cpu.py` 已内置探测：环境不可用时给出明确原因并 skip，
而不是抛出难以理解的 spawn 失败。

---

## 六、未来发展规划（按优先级）

> 以下内容**默认推迟**，仅在第七章阶段 0-4 全部完成后按明确指示实施。

### P1 — 指标 torchmetrics 迁移（low）
- **背景**：内置指标（mAP / mIoU / PQ）正确性已由测试保障，torchmetrics 作为可选后端。
- **实现步骤**：① `metrics/detection.py` 增加 `backend="torchmetrics"` 分支；② 分割/全景指标同法；
  ③ 保持 `BaseMetric` 接口不变（update/compute/reset），配置 `metrics.backend` 切换。
- **验收**：内置与 torchmetrics 后端在同一合成数据上结果一致（容差 1e-4）。

### P2 — 模型库继续扩充（medium）
- 骨干：更多 Transformer 变体（ConvNeXtV3 趋势、MoE 视觉骨干等，按最新论文跟进）。
- 检测：YOLO 系列完整化（YOLOv12 等）、DETR 家族（Co-DETR / DINOv2 集成）。
- 分割：Mask2Former 完整训练增强、SAM 类交互分割（另行立项）。
- 新增组件遵循 R1-R7，注册即用，配套合成数据配置与测试，**并纳入 7.2 的配置 E2E 门禁**。

### P3 — 训练工程增强（low）
- 自动混合精度扩展（bf16）、DeepSpeed/FSDP 大模型训练（按需）。
- 实验管理 Web 化（可选：接入 wandb/MLflow 日志后端）。

### ⏸️ 原「P0 — 多卡实验验证」→ 已移至 7.8（环境阻塞）

---

## 七、v1.1 稳定性与工程优化计划（修订版 v2）

### 7.0 执行原则（本版新增，强制）

> v1 版计划（commit `e20d5e9`）的框架成立，但存在 4 条不成立的论断，且在执行中把绿色基线改成了红色。
> 修订版新增以下强制约束，用于消除同类问题。

1. **证据先行**：每条论断必须附 `file:line` 证据与「已核实 / 未核实」标记。
   v1 版 14 条论断中 4 条不成立（见 7.1.1），根因就是缺少这一约束。
2. **度量先于修复**：先建立能自动暴露缺陷的门禁，再改代码。
   v1 版是在「286 测试全绿 + 2 个官方配置崩溃」的情况下推进的，说明测试网没盖住装配层。
3. **测试退化防护**：禁止用新测试替换既有 `def test_*` 的函数头、禁止用缩进把测试体变成局部代码；
   CI 校验 collected 测试数不低于基线（当前基线 **405**，见第五章实测表）。
   v1 版执行中已实际静默丢失 4 条回归测试。
4. **行为变更登记**：任何改变训练语义或指标口径的改动，必须在第八章登记并说明影响（历史 checkpoint 可比性）。
5. **提交粒度**：一项任务 = 实现 + 回归测试 + 文档 + lint，一次提交；禁止跨任务混合提交。
6. **验收以官方配置为准**：每个阶段的门禁都包含「19 个官方配置 E2E」，不允许只做单点组件验收。
7. **非本计划目标**：不追求新增模型家族、不追求覆盖率数字好看，只保证已有能力真的可用。

### 7.1 v1.0.0 论断核实表

> 核实基准：git HEAD `e20d5e9`，环境 `conda env dvisionix`。v1 版 7.1 共 14 条，另补漏 6 条。

#### 7.1.1 更正记录：v1 版论断中不成立的部分

| v1 版论断 | 核实结论 | 证据 |
|---|---|---|
| `registry.py:99` 把构造参数 `name` 与注册名混用，模型名称「可能被静默丢弃」 | **❌ 误诊**。`cfg.pop("type", None) or cfg.pop("name", None)` 中 `or` 短路：`type` 存在时右侧**根本不求值**，`name` 从未被弹出，一直原样传给构造函数 | `registry.py:99`(v1.0.0)；`MODELS.build({"type":"timm_backbone","name":"resnet18"})` 实测正常 |
| EarlyStopping 的 best value / 等待计数未持久化 | **❌ 误诊**。早已实现并有测试覆盖（仅 `best_weights` 未持久化） | `callbacks/early_stopping.py:61-73`；`trainer.py:561-563`；`tests/test_training/test_stage3.py:96-123` |
| `pyproject.toml` 也参与依赖/版本多源维护 | **❌ 归因错误**。`pyproject.toml` **没有 `[project]` 表**，不持有依赖与版本；依赖只有 `setup.py` ↔ `requirements.txt` 双源 | `pyproject.toml`（仅 build-system + 工具配置） |
| 「当前环境未安装 pytest，无法复现测试数字」 | **❌ 过时**。conda 环境 `dvisionix` 中 pytest 9.1.1 可用，全量测试 69.4s 跑完，且与 README 数字**完全一致** | 实测 `286 passed, 2 skipped`（288 collected） |
| v1 版 P0-3 方案：「`name` 只作构造参数，注册键改用 `_name_`」 | **❌ 方案错误**。`name` 作为选择器是项目既有约定：3 个内置默认配置 + 3 个官方 demo 配置**全部依赖它**；按 v1 方案执行会直接打断 4 个官方配置 | `dvisionix/config/defaults/{classification,detection,segmentation}.yaml:7`；`configs/{classification,detection,segmentation}/demo_synthetic.yaml` |

> **结论：v1 版 P0-3 作废**，改为 7.3 的 1-1「`type` 为标准选择器 + `name` 兜底并告警」。

#### 7.1.2 v1.0.0 成立的真实缺陷（核实通过）

| # | 缺陷 | 证据 | 严重度 |
|---|---|---|---|
| D1 | 检测 `decode()` 契约不统一：`centernet` / `nmsfree_yolo` 缺 `iou_threshold`，而 `DetectionTask` 无条件传入 → 验证阶段必崩。R4 声称的「decode 统一契约」（v0.7.1）实际 2/11 违约 | `models/detectors/centernet.py:38-44`、`models/detectors/nmsfree_yolo.py:40-45` vs `training/tasks/detection.py:72-78` | **高** |
| D2 | `yolov10_synthetic.yaml:36` 传 `topk: 13` 给 `OneToOneYOLOLoss`，后者 `__init__` 无该参数（`topk` 属 YOLOv9/v11 的 TaskAligned，本损失是 top-1 匹配） | `configs/detection/yolov10_synthetic.yaml:36` vs `models/losses/detection/losses.py:576-583` | **高** |
| D3 | `simclr` 不在 `TASK_TYPES`，schema 阶段即失败；且修复后暴露第二层缺陷：合成数据无 simclr 分支 | `config/schema.py:63`；`tools/train.py:82-101` | **高** |
| D4 | DINO 硬编码 `feats[0].shape * 4` 推断图像尺寸。官方配置为 3 个 stride=2 stage → `feats[0]` stride=2，`*4` 得到 128 而真值 64，**2 倍误差**；`DINOLoss` 却用真值 → 同一 `bbox_embed` 在去噪分支与主分支被训练在两个坐标系 | `models/heads/detection/dino.py:121`；`configs/detection/dino_synthetic.yaml:13-16`；`losses.py:854,871-888` | **高** |
| D5 | DDP 聚合对嵌套 tuple 使用 `extend`，检测的 `(boxes, scores, labels)` 3-tuple 被拍平成 6-tuple，`update_metrics` 解包必错 | `training/trainer.py` `_concat_objects`(v1.0.0) | 高 |
| D6 | FCOS/YOLO 回归损失重复累加：`use_giou=True` 时为 GIoU+L1，`use_giou=False` 时为 **2×L1**，且配置语义与实际训练信号不一致 | `models/losses/detection/losses.py:237,460`(v1.0.0) | 高 |
| D7 | 检测回归损失未按 `num_pos` 归一，loss 量级随 batch size 与层数漂移，不同 batch size 的 `val_loss` 不可比 | `models/losses/detection/losses.py:242-246` | 中高 |
| D8 | `Trainer.validate()` 无 DDP 分支，多卡下只统计 rank0 分片；且与训练中验证是两套重复实现，未共用 evaluator | `training/trainer.py:431-437` vs `478-509` | 中高 |
| D9 | 检测管线先 resize 到目标尺寸再执行同尺寸 crop，随机裁剪完全失效（无偏移） | `data/transforms/__init__.py:123-131` | 中 |
| D10 | `ToTensor` 可能把 mask 转成 float，破坏 `CrossEntropyLoss` 的 long dtype 契约 | `data/transforms/image.py:175-189` | 中高 |
| D11 | 首张图片无预测 mask 时，mask AP 的目标尺寸退化为 `(1, 1)` | `training/evaluation.py:94-100` | 中 |
| D12 | ONNX 导出：嵌套输出不支持、verify 未 `detach().cpu()`、dynamo 未传 names/axes；且 `__init__` 即原地修改调用者模型的 device 与 train/eval 状态且从不恢复 | `export/onnx_exporter.py:29-40,71,150-176,253-273` | 中 |
| D13 | 合成数据的训练集与验证集共用 `cache_dir` 与文件名序列 → val 的前 N 张就是 train 的前 N 张，**验证集是训练集子集** | `tools/train.py:123-129` | 中 |
| D14 | 打包缺 `package_data` / `include_package_data` / `MANIFEST.in`，`dvisionix/config/defaults/*.yaml` 不进 wheel；`Config.from_default()` 从安装目录读取 → 非 editable 安装**必然失败**（且抛误导性 `ValueError`）。另 `find_packages()` 会把 `tests*` 打进 wheel | `setup.py:23`；`dvisionix/config/config.py:76-82` | 中 |
| D15 | 依赖与版本号双源且已漂移：`timm>=0.16.0` vs `>=0.9.0`、`pillow` 仅一处声明；版本号 `setup.py:18` vs `dvisionix/__init__.py:26` | `setup.py:24-48` ↔ `requirements.txt:3-29` | 低 |
| D16 | CI 仅 4 步（无 mypy / coverage / pip-audit / wheel 安装），仅测 3.10 与 3.11；无 marker 注册、无 `tests/conftest.py`；actions 未固定 SHA，无 `permissions` / `timeout-minutes` / `concurrency` | `.github/workflows/ci.yml`（28 行） | 中 |
| D17 | **EMA / DistillCallback 从配置入口不可达**：`build_callbacks` 只构造 `ModelCheckpoint` + `EarlyStopping`，而 README 与本文件第三章均将其列为已实现能力 | `training/builder.py:14-43`；`EMA(` / `DistillCallback(` 仅出现在 `tests/` | 中 |
| D18 | 任务类型常量存在三份且不一致：`tools/train.py:46` 那份还是**从未被引用的死常量**；`models/base.py:16` 不含 `simclr` | `config/schema.py:63`、`tools/train.py:46`、`models/base.py:16` | 中 |
| D19 | docstring 机械损坏 3 处：`` `name` `` 被切成「反引号 + 换行 + ame」 | `registry.py:93`、`data/base.py:17`、`metrics/base.py:16` | 低 |
| D20 | 测试只「构建模型」不「运行配置」，造成配置已覆盖的假象：`test_new_detection_configs_load` 点名了 yolov10 与 centernet，却只做 `build_model()`，从不构建 loss、从不跑 `validation_step`，因此 D1/D2 全部漏网 | `tests/test_models/test_v010_direction3.py:130-136`（另有 4 处同模式） | 中 |

> **修复进度（截至阶段 2）**：D1、D2、D3、D5、D6、D7、D8、D9、D10、D11、D13、D17、D18、D19、D20
> 共 15 项**已修复并附回归测试**；
> 剩余 5 项按计划归属：D4 与 D12 → 阶段 3（7.5），D14 / D15 / D16 → 阶段 4（7.6）。

#### 7.1.3 诚实评价：v1.0.0 的真实成色

- **成立的部分**：分层调用规则 R1–R7、Registry 注册即用、组合器目录化、10 篇专题文档、286 条测试与 69 秒的反馈速度，是扎实且少见的基线。
- **不成立的部分**：「v1.0.0 收尾 / 全库审查通过」的结论不成立。19 个官方配置有 3 个跑不通；`decode()` 契约声称统一实则 2/11 违约；docstring 仍有 3 处机械损坏；EMA/蒸馏有文档承诺但配置入口不可达。
- **根因（最重要的一条）**：**测试只覆盖组件，不覆盖装配**。所有「配置加载」类测试都止步于 `build_model()`，从不构建 loss、从不执行 `validation_step`、从不跑训练循环。所以组件级 286 条测试全绿，而装配级 3 个官方配置崩溃。这正是 7.2 阶段 0 存在的理由。

### 7.2 ✅ 阶段 0：冻结基线 + 建立门禁（已完成）

> 目标：把「哪些配置是坏的」从人肉发现变成 CI 自动告知。**此阶段不修任何产品缺陷。**

| 步骤 | 内容 | 交付物 | 状态 / 验收 |
|---|---|---|---|
| 0-1 | 固化基线快照：测试数、lint 状态、19 配置结果表 | 本文件第五章实测表 | ✅ 已记录（`286 passed, 2 skipped` / 14 PASS、3 FAIL） |
| 0-2 | **配置 E2E 门禁**：参数化遍历 `configs/**/*.yaml`（白名单跳过 `hparam_search` / `linear_eval` 两个模板），各跑 1 epoch，断言产物 `history.csv` 与 `config.resolved.yaml` 存在 | `tests/test_e2e_configs.py` | ✅ 17 条参数化用例全绿；`KNOWN_BROKEN` 临时豁免机制保留且现为空集（登记项带 `xfail(strict=True)`，修复后自动变 XPASS 失败，强制清除） |
| 0-3 | **模型契约门禁**：参数化遍历所有已注册 detector，断言 `decode()` 接受统一 kwargs 集合与前两个位置参数 | `tests/test_model_contracts.py` | ✅ 建成即精确命中 D1：`CenterNetDetector` 与 `NMSFreeYOLODetector` 缺 `iou_threshold` |
| 0-4 | **测试退化防护**：禁止替换既有测试函数头；collected 数不低于基线 | 7.0 原则 3 + 本表基线快照 | 🟡 原则已入册、基线已记录；**CI 强制校验待阶段 4-2 落地** |
| 0-5 | `tests/conftest.py` + 注册 `unit/integration/slow/cuda/ddp` marker + `--strict-markers` | `tests/conftest.py`、`pyproject.toml` | ✅ `--strict-markers` 已入 `addopts`，全量测试通过 |

### 7.3 ✅ 阶段 1：修复跑不通的官方配置（已完成）

| 步骤 | 涉及文件 | 实施 | 状态 / 验收 |
|---|---|---|---|
| 1-1 **Registry 选择器定案**（替代 v1 版 P0-3） | `registry.py` + 6 个配置 + `tests/test_registry/test_registry.py` | `type` 为标准选择器；`type` 缺失时保留 `name` 兜底并抛 `DeprecationWarning`；`_name_` 保留为显式别名；仅在 `name` 值确实是已注册键时才作为选择器消费，否则原样留给构造函数。同时把 3 个 defaults + 3 个 demo 配置迁移到 `type` | ✅ 6 个配置迁移后全绿；旧 `name`-only 配置仍可用且有弃用告警；新增 5 条测试覆盖 `type` 优先级、`_name_` 别名、legacy 告警、未注册 name 不被消费、`timm_backbone(name="resnet18")` 真造 resnet18 |
| 1-2 **SimCLR 端到端**（把 v1 版 P0-2 修完） | `config/schema.py`、`tools/train.py`、`tests/` | ① `TASK_TYPES` 加 `simclr`；② **同时**给 `build_synthetic_dataset` 补 simclr 分支（只返回 image，由 `SimCLRTransforms` 生成双视角），并对未知 task_type 显式报错；③ 任务类型常量收敛：以 `config/schema.py:TASK_TYPES` 为**唯一权威来源**（`tools/train.py` 的死常量 `_TASK_TYPES` 已删除），一致性由测试 `test_task_mapping_covers_all_task_types` 守护；④ `training.ema` 加入已知键并校验 `decay ∈ (0,1)` | ✅ 该配置 CLI 完成 1 epoch（exit 0） |
| 1-3 **CenterNet decode 契约** | `models/detectors/centernet.py` | `decode()` 增加 `iou_threshold: float = 0.5`；热图峰值检测不需要 NMS，参数接受并忽略，docstring 说明原因 | ✅ 契约测试通过；`centernet_synthetic.yaml` 完成 1 epoch |
| 1-4 **YOLOv10 双重缺陷** | `configs/detection/yolov10_synthetic.yaml`、`models/detectors/nmsfree_yolo.py` | ① `OneToOneYOLOLoss` 是 top-1 贪心匹配，`topk` 属 YOLOv9/v11 遗留 → **从配置移除**并在原处留注释防止再次误加；② 修 `nmsfree_yolo` 的 decode 契约（同 1-3） | ✅ 契约测试通过；`yolov10_synthetic.yaml` 完成 1 epoch |
| 1-5 **卫生清理** | `registry.py`、`data/base.py`、`metrics/base.py`、`data/datasets/custom.py`、`training/builder.py`、`README.md` | ① 修 3 处损坏 docstring；② `CustomDataset` 去掉 `staticmethod` 包装并补构造期 `callable()` 校验（显式 `TypeError`，非 `assert`）；③ D17 选择「**把 EMA 接通配置**」：`build_callbacks` 支持 `training.ema.{enabled,decay,swap_for_validation,decay_warmup_epochs,save_final}`，默认关闭；蒸馏因需要教师权重实例，**明确降级为编程式 API** 并同步修正 README 与本文件第三章 | ✅ 全文无损坏 docstring；新增 4 条 EMA 配置测试 + 4 条 collate 测试；README 能力声明与配置入口一致 |

**阶段 1 门禁达成情况**：17 配置 E2E 全绿 ✅、契约测试全绿 ✅、`xfail` 清空 ✅、
`ruff`/`black` 全绿 ✅、collected 由 288 升至 344 ✅。

### 7.4 ✅ 阶段 2：P0 正确性（已完成）

| 步骤 | 状态 | 实际实施与验收证据 |
|---|---|---|
| 2-1 **DDP 递归聚合**（重做 D5 + 修 D8） | ✅ | `_concat_objects` 改为递归：Tensor 沿 batch 拼接、list 逐元素、**tuple 按位置递归**（不再被 `extend` 拍平）、dict 按 key；结构不一致显式报错；新增 `_gather_preds_targets` 共享 helper 与 `_update_metrics_for_step` / `_accumulate_step_logs`，训练中验证与 `validate()` 共用同一实现。`validate()` 补上 DDP 分支。新增 `tests/test_training/test_ddp_cpu.py`（**CPU gloo 双进程**）：单进程 vs 双进程全局指标在 `1e-6` 内一致，且断言全局指标 ≠ rank0 分片指标。修复 CPU 分布式路径的 3 处阻塞（设备解析强制 `cuda:*`、`device_ids=[None]`、`strategy="auto"` 要求 CUDA） |
| 2-2 **损失语义**（重做 D6 + 修 D7） | ✅ | ① 引入显式 `giou_weight` / `l1_weight`（与同文件 **`DETRLoss`** 的 `bbox_weight`/`giou_weight` 约定一致）；旧 `use_giou` 按**字面意图**迁移并告警，与显式权重同时给出则报错；② 各分量先乘回正样本数还原为「和」，循环末尾统一除以 `num_pos` / `num_cls_terms`，回归与 cls 均成为真实均值；③ 新增 14 条测试锁定三模式可区分、`combined == giou + l1`（证明各只算一次）、权重线性、batch size 不变性、空目标无 NaN |
| 2-3 **几何变换契约**（修 D9） | ✅ | ① 检测训练管线改为先 resize 到 1.1× 再随机 crop（与分类预设一致），并注释原因；② `RandomCrop` / `CenterCrop` / `BoxSyncRandomCrop` 统一增加 `on_small=error\|pad\|resize`，**默认 `error`**，终止 v1.0.0 的静默错误尺寸行为；③ 三种几何变换入口校验 boxes 与 labels 数量一致。新增 15 条测试，含「裁剪偏移必须非零且随种子变化」与 image/boxes/mask 同步 |
| 2-4 **mask dtype 与 mask AP**（修 D10 + D11） | ✅ | ① `ToTensor` 对 mask **与 ndim 无关**地短路为 long；② `MaskToTensor` 无论输入类型强制 long，并拦截非整值浮点、负值、非法形状；③ `evaluate_mask_ap` 空预测时退化为**输入图像尺寸**而非 `(1, 1)`，并清理函数内重复 import。新增 11 条 dtype 契约测试 + 1 条「首图空预测」回归测试（用 spy 指标断言 GT 尺寸） |
| 2-5 **合成数据泄漏**（修 D13） | ✅ | `build_synthetic_dataset` 增加 `split` 参数，文件名加划分前缀（`train_img_0000.png` / `val_img_0000.png`），并改用按 `(split, i)` 派生的稳定随机种子使数据可复现；检测框生成补上小尺寸防护。新增 6 条测试：4 类任务的 train/val 路径无交集、可复现性、未知 task_type 报错 |

**阶段 2 门禁达成情况**：全量测试 `403 passed, 2 skipped`（collected 405）✅、
17 配置 E2E 全绿 ✅、`ruff`/`black` 全绿 ✅、CPU gloo 双进程一致性 ✅。

**行为变更**：2-2（损失语义与归一化）、2-3（检测裁剪增强真实生效）、2-5（合成数据口径）
均已登记入第八章「行为变更登记」。

### 7.5 阶段 3：训练 / 评估 / 导出闭环（3–4 天）

| 步骤 | 涉及文件 | 实施 | 验收 |
|---|---|---|---|
| 3-1 **DINO 尺寸来源**（修 D4） | `heads/detection/dino.py`、`detectors/dino.py`、`training/tasks/` | `image_hw` 由 task 显式传入 head（或 head 直接取 `batch["image"].shape[-2:]`，head 已收到 batch）；`*4` 仅保留为 stride 感知的兼容 fallback | 去除硬编码 `*4` 主路径；stride=2 / stride=4 / 非方形输入下去噪目标坐标正确；**行为变更登记** |
| 3-2 **统一 Evaluator**（修 D8） | `training/trainer.py`、`training/evaluation.py` | 抽出 `Evaluator`，训练中验证与独立 `validate()` 共用；`validate()` 补 DDP 分支（`DistributedSampler` + 复用 gather helper）；放弃启发式 batch-size 推断，回归 `int(batch["image"].shape[0])` | 独立 `validate()` 与训练中指标一致；DDP 下 `validate()` 给出全局指标而非 rank0 分片 |
| 3-3 **checkpoint 契约** | `training/{trainer,checkpoint,workdir}.py` | checkpoint 增加 `schema_version` / `task_type` / `config_hash`（复用 `workdir.py:57-64` 的 `hash_config`）；resume 默认拒绝配置不匹配并提供显式 override；修正梯度累积尾窗口分母（尾部不足一窗时不应继续除以 `accumulate_grad_batches`） | resume 前后 optimizer / scheduler / RNG / EMA / early-stopping 一致；5 batch + accum=2 的尾窗梯度正确 |
| 3-4 **ONNX 契约**（修 D12） | `export/onnx_exporter.py`、tests | 递归 flatten（支持 dict/list/tuple 任意嵌套）并落盘输出路径映射；verify 统一 `detach().cpu().numpy()`；dynamo 路径传 `input_names`/`output_names`/`dynamic_axes`，不支持时显式报错；**用 `try/finally` 保存并恢复模型 device 与 train/eval 状态**；清理未用导入 | 分类 / 分割 / 多尺度检测均可导出；动态 batch 可执行；导出器不改变原模型状态；ONNX Runtime 数值验证 |

### 7.6 阶段 4：打包与 CI（2–3 天）

| 步骤 | 涉及文件 | 实施 | 验收 |
|---|---|---|---|
| 4-1 **打包一致性**（修 D14 + D15） | `pyproject.toml`、`setup.py`、`requirements.txt`、`MANIFEST.in` | ① 迁移到 `pyproject.toml [project]` 单源，消除依赖/版本双源；② `[tool.setuptools.package-data]` 纳入 `dvisionix/config/defaults/*.yaml`；③ `find_packages(exclude=["tests*"])`；④ 补 `LICENSE` 与 `CHANGELOG`；⑤ 统一版本号来源；⑥ 依赖拆分 core / dev / export / datasets extras | 构建 wheel 后在新虚拟环境安装，`Config.from_default()` 与最小 import 均成功；**该测试纳入 CI** |
| 4-2 **真实测试门禁**（修 D16 + D20） | `.github/workflows/ci.yml`、`pyproject.toml`、`tests/conftest.py` | CI 顺序：锁定依赖 → ruff → black → mypy（先修 `pyproject.toml:24` 的 `python_version="3.8"` 到 3.10）→ 单测 → **配置 E2E** → coverage（初始 60%，稳定后 75%）→ wheel 安装 → pip-audit；矩阵补 3.12；启用 strict markers | PR 可直接看到测试、覆盖率、类型、打包、安全结果；测试状态以 CI 为准 |
| 4-3 **安全与发布** | CI、`dependabot.yml`、release workflow、checkpoint loader | pip-audit + Dependabot；actions 固定 SHA + `permissions: contents: read` + `timeout-minutes` + `concurrency`；建立 tag → build → smoke → TestPyPI/PyPI 流程；checkpoint 默认 `weights_only=True`，完整 pickle 需显式声明可信 | 发布 wheel 可在干净环境运行；不可信 checkpoint 风险与 API 策略有文档 |
| 4-4 **文档与解释器对齐** | `README.md`、`setup.py` | README 的测试数字改为「以 CI 为准」并给当前实测值；`setup.py:57-60` classifier 补 3.14，或在 README 明确标注 3.14 为实验性 | 文档声明与 CI 矩阵、`python_requires` 三者一致 |

### 7.7 阶段 5：性能（P2，最后）

> 先建立 benchmark，再修改实现。阶段 0-4 未完成前不启动。

1. **EMA 与数据搬运**：shadow tensor 原地更新；区分浮点参数与非浮点 buffer；DataLoader 暴露 `pin_memory` /
   `persistent_workers` / `prefetch_factor`；支持 `non_blocking` 设备搬运。
2. **DDP 评估通信**：保留 `all_gather_object` 作为 fallback；Tensor 结果改用 padding + valid count + `dist.all_gather`；
   大型检测评估支持 rank 分片落盘、rank0 汇总；记录通信耗时与样本数。
3. **mAP / PQ / matcher / assigner**：mAP 复用排序与 IoU 中间结果；PQ 使用类别过滤、bbox 粗筛与分块 overlap，
   禁止超大 `(P,G,H,W)` 中间张量；matcher 减少 GPU→CPU 同步；向量化 TaskAlignedAssigner 与 OneToOneYOLO 的 Python 循环。
4. **运行时契约与 API**：实现或删除 `Sample._KNOWN_KEYS` 承诺；明确 RGB/BGR 并让 `ImageMode` 参与校验；
   让 `provides_normalization` 真正阻止重复归一化；`BaseModel.get_device()` 处理无参数模型；
   `assert` 改显式异常、`out_indices` 越界禁止静默取模；`MetricCollection` 对重复名称报错；
   库代码 `print()` 统一为 logger；TensorBoard 标量真正接入 Trainer 生命周期；减轻顶层 import。

**验收**：优化前后指标在容差内一致；EMA 开销低于训练总耗时 3%，或提供无法达到时的基准说明；
高分辨率全景评估无异常内存峰值；每项优化都有固定输入的前后 benchmark。

### 7.8 环境阻塞项

| 项目 | 阻塞原因 | 替代方案 | 解除条件 |
|---|---|---|---|
| 多卡 NCCL 验证（原第六章 P0） | 当前环境 `torch.cuda.is_available() == False` | CPU `gloo` 双进程一致性测试（已纳入 7.4 的 2-1），CI 可长期运行 | 获得 2+ GPU 环境后执行 `torchrun --nproc_per_node=2 tools/train.py --config ... --devices 0,1`，验证各 rank 批数一致、指标 all_gather 无死锁、checkpoint 仅 rank0 保存 |

### 7.9 Definition of Done（强化版）

任务只有同时满足以下全部条件才可标记为完成：

1. 实现已提交，没有通过静默 fallback 掩盖错误。
2. 至少有一条针对原缺陷的回归测试，且 **collected 测试总数不低于当前基线（405）**。
3. 相关单元与集成测试通过，**17 个可训练官方配置 E2E 全绿**。
4. `ruff` 与 `black --check` 全绿。
5. 文档、配置示例与 API 行为一致（含 README 的能力声明）。
6. 性能改动有固定输入的前后 benchmark。
7. 分布式改动有 CPU `gloo` 单卡 / 双进程一致性证据。
8. 发布改动有干净环境 wheel 安装证据。
9. 凡改变训练语义或指标口径的改动，已在第八章「行为变更登记」中登记。

---

## 八、版本记录

### 版本里程碑（精简）

| 版本 | 里程碑 |
|---|---|
| v1.1（进行中） | 稳定性与工程优化：阶段 0-2 已完成（门禁 + 3 个官方配置 + 15 项缺陷），阶段 3-5 待办 |
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

### 行为变更登记

> 凡改变训练语义或指标口径的改动必须在此登记，说明影响面与历史可比性。

| 版本 | 变更 | 影响 |
|---|---|---|
| v1.1（已实施） | FCOS / YOLO 回归损失改为显式 `giou_weight` / `l1_weight`，默认 **GIoU only**（`giou_weight=1.0, l1_weight=0.0`）。旧 `use_giou` 按字面意图迁移（`True`→GIoU、`False`→L1）并发出 `DeprecationWarning`；与显式权重同时给出则报错（对应 7.4 的 2-2） | **训练语义变更**：v1.0.0 中 `use_giou=True` 实为 **GIoU+L1**、`False` 实为 **2×L1**。修正后回归项减少一个 L1 分量，loss 量级与最优学习率会变化，**v1.0.0 的历史 checkpoint 不可直接续训**，既有超参需重新标定 |
| v1.1（已实施） | 检测损失各分量改为真实均值：`reg` / `center` 除以 `num_pos`，`cls` 除以 `(图, 层)` 项数（对应 7.4 的 2-2） | **指标口径变更**：loss 不再随 batch size 与特征层数漂移，但数值与 v1.0.0 不可直接比较，历史 `best` 判据失效。这使得不同 batch size 的 `val_loss` 首次可比 |
| v1.1（已实施） | 检测训练管线改为先 resize 到 1.1× 再随机裁剪（对应 7.4 的 2-3） | **训练语义变更**：v1.0.0 的「resize 到目标尺寸 → 同尺寸裁剪」使随机裁剪退化为恒等操作，**增强完全未生效**；修正后训练数据分布改变，等价于引入了此前缺失的数据增强 |
| v1.1（已实施） | `RandomCrop` / `CenterCrop` / `BoxSyncRandomCrop` 在输入小于目标尺寸时**默认直接报错**（`on_small="error"`）（对应 7.4 的 2-3） | **行为变更**：v1.0.0 是静默返回错误尺寸（`CenterCrop` 经负索引切片返回更小的图）。依赖旧行为的自定义管线需显式传 `on_small="pad"`。内置预设管线不受影响（resize 后尺寸恒满足） |
| v1.1（已实施） | 合成数据 train / val 使用不同 split 前缀与独立随机种子（对应 7.4 的 2-5） | **指标口径变更**：v1.0.0 的 val 前 N 张就是 train 前 N 张，demo 指标含数据泄漏、偏乐观；修正后 `val_loss` / `val_acc` 会明显变差，属预期修正。同时合成数据变为可复现（原为每次运行随机） |
| v1.1（待实施） | DINO 去噪目标改用真实 `image_hw`（对应 7.5 的 3-1） | **训练语义变更**：去噪分支坐标尺度修正（官方配置下原为 2 倍误差），需重新训练评估效果 |
