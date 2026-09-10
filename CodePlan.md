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

| 指标 | v1.0.0 基线（HEAD `e20d5e9`） | 当前（阶段 0-5 全部完成后） |
|---|---|---|
| 测试 | `286 passed, 2 skipped`（69.4s），collected = 288 | `481 passed, 2 skipped`（114.5s），collected = 483 |
| 测试构成 | 纯组件级 | 组件级 + 配置 E2E 门禁 24 + 契约/回归/CPU-DDP/导出/打包/性能 共 195 |
| 静态检查 | `ruff` / `black --check` 全绿 | `ruff` / `black --check` 全绿；`mypy` 可运行（基线 121 errors，非阻断） |
| 覆盖率 | 未统计 | **91%**（CI 门禁 ≥90%） |
| 官方配置 E2E（19 个各跑 1 epoch） | **14 PASS / 3 FAIL / 2 模板不可跑** | **17 PASS / 0 FAIL / 2 模板不可跑** |
| 跑不通的配置 | `classification/simclr_synthetic`、`detection/centernet_synthetic`、`detection/yolov10_synthetic` | 无 |
| 不可跑的模板（非缺陷） | `classification/hparam_search`（用 `tools/hparam_search.py`）、`classification/linear_eval`（占位 checkpoint 路径） | 同左 |
| CPU 双进程 DDP 一致性 | 无此测试（`test_ddp_smoke` 需 2+ GPU，恒跳过） | ✅ 单进程 vs 2 进程全局指标在 `1e-6` 内一致 |
| 导出契约 | 嵌套输出崩溃、导出改动调用者模型 | ✅ 嵌套输出可导出并数值验证、导出不改变原模型状态 |
| 发行物 | wheel 缺内置默认配置、混入 `tests/` | ✅ `config/defaults/*.yaml` 全部在包内、无 `tests/`；干净安装后 `Config.from_default()` 可用 |
| 测试顺序依赖 | 未知（依赖顶层 eager import） | ✅ **56 个测试文件逐个独立运行全部通过** |

### 📈 性能基准（`tools/benchmark.py`，CPU / torch 2.12）

| 场景 | v1.0.0 | v1.1 | 倍数 |
|---|---|---|---|
| 匈牙利匹配（300 查询 × 30 GT） | 13 652 ms | **18.6 ms** | ~734× |
| 全景质量 PQ（512×512 / 25 实例） | 582 ms | **123 ms** | ~4.7× |
| EMA 开销（占训练总耗时） | 17.11% | **1.44%** | 目标 <3% ✅ |
| 只导入 config | ~3–4 s | **0.10 s** | 顶层 import 惰性化 |
| mAP（100 图 × 20 框） | 173 ms | 180 ms | 未优化（实测非瓶颈） |
| TaskAligned 分配 | 1.1 ms | 1.0 ms | 未优化（实测非瓶颈） |

> 说明：测试数从 288 增至 457 全部是**新增门禁与回归测试**，不是实现膨胀；
> 其中 24 条为配置端到端门禁、2 条为 CPU gloo 双进程、3 条为打包契约、
> 其余为契约与回归测试。无既有测试被删除；仅 1 条既有断言随契约修正同步更新
> （`test_task_config.py` 的 `accuracy` → `val_accuracy`，并补上「不得为 0」的断言）。
>
> 耗时说明：表中当前值 114.5s 为空载实测；并发执行其他命令时会到 150–190s，
> 差异来自 CPU 争用，并非测试本身变化。

### 🔄 当前状态：v1.1 稳定性与工程优化

- **第七章是本项目唯一执行计划。**
- **阶段 0-5 已全部完成**（门禁 → 配置修复 → P0 正确性 → 训练/评估/导出闭环 →
  打包与 CI → 性能优化），见 7.2 / 7.3 / 7.4 / 7.5 / 7.6 / 7.7 的完成标记。
  20 项已核实缺陷 D1-D20 全部修复并附回归测试；官方配置由 14/17 可跑提升到 **17/17 全绿**。
- **v1.1 的开发工作到此结束**，可以进入第六章的 P1/P2/P3（模型库继续扩充等），
  或按需把 mypy / pip-audit 从信息性门禁升级为阻断（见 7.6 的升级条件）。
- **v1.1 早期一批未提交改动已整体回退**（回退原因与更正见 7.1.1，
  补丁留档于 `.dev_archive/wip-v1.1-partial.patch`，已在 `.gitignore` 中排除）。
- 该批次中**诊断正确且实现无误**的两项（DDP 递归聚合、FCOS/YOLO 重复累加 L1）
  已在阶段 2 重新实现，并补齐其缺失的归一化与测试。
- **用户文档**：新增 [docs/v1.1_changes.md](docs/v1.1_changes.md)（迁移指南），
  并为 7 篇专题文档补充「v1.1 变更要点」章节、更新文档索引与 README。

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

> 以下内容**默认推迟**，仅在第七章阶段 0-4 全部完成（现已达成）后按明确指示实施。

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
   CI 校验 collected 测试数不低于基线（当前基线 **483**，见第五章实测表）。
   v1 版执行中已实际静默丢失 4 条回归测试。
8. **测试必须自播种**：「loss 下降」这类断言要在**构建模型之前**播种
   （模型初始权重也是随机的），否则会依赖此前测试消耗掉的全局 RNG 状态，
   表现为顺序相关的偶发失败 —— 阶段 3 已实际遇到一次。
9. **单一事实来源**：依赖、版本、任务类型常量、decode 契约等都只能有一处定义；
   其余位置要么引用它，要么由测试守护一致性（阶段 1 与阶段 4 各消除了一处双源）。
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

> **修复进度（截至阶段 4）**：**D1-D20 共 20 项全部已修复并附回归测试**。
> 其中 mypy 与 pip-audit 两项 CI 门禁目前为**信息性**（原因见 7.6 的 4-2 说明），
> 它们不属于 D 编号缺陷，而是门禁成熟度的后续工作。

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

### 7.5 ✅ 阶段 3：训练 / 评估 / 导出闭环（已完成）

| 步骤 | 状态 | 实际实施与验收证据 |
|---|---|---|
| 3-1 **DINO 尺寸来源**（修 D4） | ✅ | `DINODetrHead` 新增 `_resolve_image_hw`：优先取 `batch["image"].shape[-2:]`（`needs_batch=True`，训练时 batch 一定带 image）；其次用显式配置的 `out_stride` 推断；**两者皆无则直接报错**，不再静默猜尺寸。硬编码 `*4` 主路径已删除。新增 `tests/test_models/test_dino_image_hw.py`：stride=2 / stride=4 / 非方形输入下去噪目标必须等于按真实尺寸归一化的结果，并额外断言「按 4 倍特征图推断会得到不同结果」以确保该测试真能区分对错 |
| 3-2 **统一 Evaluator**（修 D8） | ✅ | 抽出 `Trainer._evaluate(loader)`，训练中验证与独立 `validate()` 共用同一实现；`validate()` 因此获得：DDP 全局指标（不再只算 rank0）、`on_validation_begin/end` 回调（**EMA 权重交换依赖它**，v1.0.0 的 `validate()` 报的是未交换权重的指标）、统一的 `reset_metrics` 生命周期。删除 `_infer_batch_size` 的启发式（原实现会在 `preds=(Tensor,Tensor)` 时把 tuple 长度当 batch size，并以 `except: return 1` 静默兜底），改为只认 `image`/`image1`/`image2` 且取不到就报错。另修正 `fit()` 里**重复调用** `on_validation_epoch_end()` 的问题（第二次在已 reset 的累加器上算出全 0，正是 history.csv 里那列裸 `accuracy` 的来历），epoch 级指标现统一以 `val_` 前缀并入 | 
| 3-3 **checkpoint 契约** | ✅ | checkpoint 增加 `schema_version`（当前 1）/ `task_type` / `model_type` / `config_hash`（复用 `hash_config`，由 `build_trainer` 注入）。resume 时校验：schema 过高直接拒绝、任务/模型类型不符直接拒绝、配置哈希不符**默认拒绝**并提供 `allow_config_mismatch=True` 显式 override、旧格式（无 schema_version）给出 `DeprecationWarning` 后放行、纯 state_dict 明确拒绝。新增 `_unwrap_module` 剥掉 DDP / `torch.compile` 包装以免类型校验在单卡多卡间误报。**梯度累积分母按所在窗口长度计算**（`_accumulation_divisor`），修掉尾窗梯度被系统性低估的问题 |
| 3-4 **ONNX 契约**（修 D12） | ✅ | ① `_flatten_outputs` 改为递归，并额外返回「输出名 → 访问路径」映射，落盘到 ONNX `metadata_props.output_path_map`；② `__init__` 不再改动调用者模型，新增 `_temporary_eval()` 上下文管理器，用 `try/finally` 恢复 device 与 train/eval 状态（**失败路径也恢复**）；③ verify 统一 `detach().cpu().numpy()`，输出数量不一致或形状不一致时显式报错而非被 `zip` 静默截断；④ dynamo 路径传 `input_names`/`output_names`，并在 `dynamic_axes` 非空时**显式报错**（dynamo 用 `dynamic_shapes`，静默忽略会让动态 batch 承诺失效）；⑤ 清理未用导入。新增 `tests/test_export/test_onnx_contract.py`（17 条）覆盖递归展平、状态保护、失败恢复、dynamo 参数契约，以及分类 / 嵌套检测输出 / 分割风格输出 / 动态 batch 的 ONNX Runtime 数值验证 |

**阶段 3 门禁达成情况**：全量测试全绿 ✅、17 配置 E2E 全绿 ✅、`ruff`/`black` 全绿 ✅。

**行为变更**：3-1（DINO 去噪坐标尺度）、3-2（history 列名与指标口径）、3-3（resume 校验与梯度累积分母）
已登记入第八章。

### 7.6 ✅ 阶段 4：打包与 CI（已完成）

| 步骤 | 状态 | 实际实施与验收证据 |
|---|---|---|
| 4-1 **打包一致性**（修 D14 + D15） | ✅ | ① 元数据全部迁移到 `pyproject.toml [project]`，**`setup.py` 已删除**；② `[tool.setuptools.package-data]` 纳入 `dvisionix/config/defaults/*.yaml`；③ 包发现改为 `include = ["dvisionix*"]` 并排除 `tests*/tools*/configs*/docs*`；④ 新增 `LICENSE` 与 `CHANGELOG.md`；⑤ 版本号单源：`dynamic = ["version"]` 从 `dvisionix.__version__` 静态读取；⑥ 依赖拆为 `export` / `models` / `datasets` / `coco` / `metrics` / `dev` / `full` 七个 extras；⑦ `requirements.txt` 降级为**指向 pyproject 的指针**，不再声明任何版本（漂移根因消除） |
| 4-1b **wheel 验证**（纳入 CI） | ✅ | 实测构建 `dvisionix-1.0.0-py3-none-any.whl`：**4 个 `config/defaults/*.yaml` 全部在包内**、`tests/` 条目为 0、`LICENSE` 位于 `dist-info/licenses/`。新增 `tests/test_packaging.py`（3 条）把「wheel 内容」与「装完能用」变成断言：构建 wheel → 校验载荷 → `pip install --no-deps --target` → 在仓库**之外**的子进程里调用 `Config.from_default()`（确保解析到安装副本而非源码树）。CI 的 `install` job 另做一次带真实依赖的干净 venv 安装 |
| 4-2 **真实测试门禁**（修 D16 + D20） | ✅ | CI 重写为 4 个 job：`quality`（ruff / black / mypy）、`test`（矩阵 3.10/3.11/3.12：快速测试 → 全量测试 + **覆盖率下限 90%** → 打包测试）、`install`（构建 wheel → 干净 venv 真实安装 → 烟雾测试 → 上传产物）、`security`（pip-audit）。全部 action 固定到 commit SHA；workflow 级最小权限 `contents: read`、`timeout-minutes`、`concurrency` 取消旧运行。`--strict-markers` 已启用；marker 与 `tests/conftest.py` 在阶段 0 落地。静态检查工具（ruff/black/mypy）**锁定版本**以保证判定可复现 |
| 4-3 **安全与发布** | ✅ | ① 新增 `.github/dependabot.yml`（pip + github-actions，按周，开发工具分组）；② 新增 `.github/workflows/release.yml`：tag → build sdist+wheel → 载荷校验 → 干净 venv 烟雾测试 → 校验和 → TestPyPI/PyPI（Trusted Publishing，`id-token: write`，两个 environment 需在 PyPI 端预先登记）；③ **checkpoint 反序列化策略**：`load_backbone` 默认 `weights_only=True`（第三方权重按不可信处理），需要 pickle 时必须显式 `trusted=True`；`Trainer.load_checkpoint` 因断点续训必须恢复优化器/回调/RNG 等非张量状态，默认 `trusted=True` 并在 docstring 写明风险，另提供 `trusted=False` 走安全模式；④ 加载前校验结构、schema 版本、任务/模型类型、配置哈希，并新增「`model_state_dict` 必须只含张量」的校验 |
| 4-4 **文档与解释器对齐** | ✅ | README：安装段补 extras 说明与「依赖唯一来源是 pyproject」；明确 `requires-python>=3.10`、CI 覆盖 3.10/3.11/3.12、3.13/3.14 为**实验性**；测试段给出当前实测数字（447 passed / 覆盖率 91%）与全部门禁命令；补 Windows 运行提示与许可证/变更日志链接。classifier 已补 `3.14`（`pyproject.toml`） |

#### 4-2 的两项「信息性门禁」说明

计划要求 CI 展示类型与安全检查结果，但这两项当前**无法立即作为阻断门禁**，原因如下（已如实记录，未用配置掩盖）：

| 项目 | 现状 | 为何暂不阻断 | 升级条件 |
|---|---|---|---|
| **mypy** | `mypy dvisionix` 报 **121 errors / 39 files**（`arg-type` 28、`override` 25、`union-attr` 17 为前三类） | 一次性修复 121 处（含 25 处 Liskov 违例，可能牵动签名）不属于本阶段范围；设为阻断会让 CI 从第一天起恒红，门禁失去意义 | 逐文件收紧并修复后改为阻断。**注意**：`[tool.mypy].python_version` 必须 ≥3.12 —— numpy≥2 的 stub 使用 PEP 695 `type` 语句，写成项目下限 3.10 会让 mypy 直接拒绝运行；项目自身下限仍由 `requires-python` 与 CI 矩阵保证 |
| **pip-audit** | 已安装项目依赖后执行 | 上游（torch / tensorflow 等）CVE 与本项目代码无关，需人工 triage 并显式 `--ignore-vuln` 记录后才适合阻断 | 建立 triage 基线后转为阻断 |

**阶段 4 门禁达成情况**：全量测试全绿 ✅、17 配置 E2E 全绿 ✅、`ruff`/`black` 全绿 ✅、
wheel 载荷与干净安装验证通过 ✅、覆盖率 91% ≥ 90% ✅。


### 7.7 ✅ 阶段 5：性能（已完成，先测后改）

> 新增 `tools/benchmark.py` 作为固定输入、可复现、可隔离测峰值内存的基准工具。
> **所有优化都先有基线数据，再改实现，最后以数值对拍确认结果未变。**

#### 5-0 基准工具

`tools/benchmark.py` 覆盖 7 个场景：`ema` / `pq` / `map` / `matcher` / `assigner` /
`dataloader` / `import`。支持 `--isolate`（每场景独立子进程，用于干净地测峰值内存）
与 `--json`（机器可读输出）。

#### 5-1 EMA 与数据搬运 ✅

| 项 | 实施 | 实测 |
|---|---|---|
| EMA 原地更新 | `shadow[k].mul_(decay).add_(v, alpha=1-decay)`，替换每步新建张量的 `decay*shadow + (1-decay)*v` | 开销 **17.11% → 1.44%**（目标 <3% ✅） |
| 缓存张量引用 | `on_train_begin` 缓存 `(shadow, param)` 引用对，不再每步调用 `model.state_dict()` | 同上（这一步才是达标的关键） |
| 只处理浮点张量 | 跳过整型 buffer（如 `num_batches_tracked`） | 数值语义更正确 |
| **resume 保留 EMA** | `on_train_begin` 检测到已恢复的 shadow 时不再用当前权重重建 | 修复「续训把 EMA 状态清零」的隐藏缺陷 |
| DataLoader 开关 | 配置暴露 `pin_memory` / `persistent_workers` / `prefetch_factor`，DDP 重包装时一并继承 | — |
| `non_blocking` 搬运 | `move_to_device(..., non_blocking=)` + `BaseTask.to_device()`，由 `training.non_blocking` 下发 | CPU 上无收益也无副作用；需 pin_memory + CUDA 才有意义 |

#### 5-2 DDP 评估通信 ✅（部分：以度量为主）

- **已落地**：评估聚合增加耗时与样本数统计，`Trainer.gather_report()` 暴露
  `calls / ms_total / ms_per_call / local_samples / ms_per_sample`，
  分布式验证时打印一次；`tests/test_training/test_ddp_cpu.py` 断言这些字段确实被记录。
- **未落地（如实记录）**：计划中的「Tensor 结果改用 padding + valid count +
  `dist.all_gather`」**没有实现**。原因：本环境是 CPU + gloo、张量规模很小，
  无法用基准证明相对 `all_gather_object` 有收益，这与本阶段「先测后改」的原则冲突；
  在没有证据的情况下引入复杂度不符合 DoD。该优化应在有 2+ GPU + NCCL 的环境上
  先用 `tools/benchmark.py` 建立通信基线再决定。

#### 5-3 mAP / PQ / matcher / assigner ✅

| 项 | 实测（前 → 后） | 数值一致性 |
|---|---|---|
| **matcher** | **13 652 ms → 18.6 ms** | 穷举对拍确认仍为最小代价；新增 11 条测试 |
| **PQ** | **582 ms → 123 ms** | 与 v1.0.0 朴素实现逐位对拍（PQ/SQ/RQ 完全一致） |
| mAP | 173 → 180 ms | 持平 |
| assigner | 1.1 → 1.0 ms | 持平 |

- **matcher**：该实现要求「行数 ≤ 列数」，v1.0.0 无论形状都把代价矩阵补成
  `max(n,m)²` 方阵再跑纯 Python O(n²m)。转置后 300×30 变成 30×300，
  规模下降两个数量级。
- **PQ**：v1.0.0 的 `p_flat[:, None, :] & g_flat[None, :, :]` 会实体化
  `(P, G, H*W)` 布尔张量（1024×1024 下约 400MB）。改为**包围盒粗筛 +
  交集区域局部逻辑与**，峰值与 `P + G` 张单通道掩码同阶。
- **mAP / assigner 未优化**：基准显示它们不是瓶颈（173 ms / 1 ms），
  按「先测后改」原则不做无依据的改动。
- 新增 `tests/test_metrics/test_panoptic_equivalence.py`（8 条）与
  `tests/test_models/test_matcher_optimality.py`（11 条），把「优化不改变数值」固定下来。

#### 5-4 运行时契约与 API ✅

| 项 | 处理 |
|---|---|
| `Sample._KNOWN_KEYS` | **实现**：新增 `Sample.unknown_keys()` 与 `EXTENDED_KEYS`；`BaseDataset` 对每个未知字段名告警一次（捕捉 `bboxes` 这类静默失效的拼写错误） |
| `provides_normalization` | **真正生效**：`TransformPipeline` 拒绝同一条流水线内出现两个归一化算子（此前该标记被聚合但从未被消费） |
| `get_device()` | 无参数模型不再抛 `StopIteration`：依次看参数、buffer，都无则返回 CPU |
| `out_indices` | 越界由**静默取模**改为显式报错（`backbones/feature.py`、`sequential.py`） |
| `assert` → 异常 | 库代码 10 处 `assert` 全部改为显式 `ValueError` / `TypeError`（`assert` 在 `python -O` 下会被剥离） |
| `MetricCollection` 重名 | 构造与 `add()` 时校验成员名唯一（重名会在汇总时静默互相覆盖） |
| `print()` → logger | `load_backbone` 的告警改走 `dvisionix.checkpoint` logger |
| TensorBoard | Trainer 每个 epoch 调用 `logger.log_metrics(..., console=False)`，标量首次真正写入 TB + JSONL |
| 顶层 import | 改为惰性（PEP 562）：`from dvisionix.config import Config` **3~4 s → 0.10 s**；注册表取用前需显式导入对应子模块，已在文档说明 |
| DDP 重复计算指标 | `fit()` 中重复调用 `on_validation_epoch_end()` 的分支已删除（第二次会在已重置的累加器上算出全 0） |
| MaskFormer 目标 mask 尺寸 | `pred_hw` 空预测时由 `None` 改为 `image_hw`，与 `evaluate_mask_ap` 的 D11 修复保持一致 |
| 文档 | 新增 [docs/v1.1_changes.md](docs/v1.1_changes.md) 迁移指南，并为 7 篇专题文档补充「v1.1 变更要点」 |

**阶段 5 门禁达成情况**：全量测试 `481 passed, 2 skipped` 全绿 ✅、
17 配置 E2E 全绿 ✅、`ruff`/`black` 全绿 ✅、
EMA 开销 1.44% < 3% ✅、所有优化均有前后基准与数值对拍 ✅、
**56 个测试文件逐个独立运行全部通过**（确认无测试顺序依赖）✅。


### 7.8 环境阻塞项

| 项目 | 阻塞原因 | 替代方案 | 解除条件 |
|---|---|---|---|
| 多卡 NCCL 验证（原第六章 P0） | 当前环境 `torch.cuda.is_available() == False` | CPU `gloo` 双进程一致性测试（已纳入 7.4 的 2-1），CI 可长期运行 | 获得 2+ GPU 环境后执行 `torchrun --nproc_per_node=2 tools/train.py --config ... --devices 0,1`，验证各 rank 批数一致、指标 all_gather 无死锁、checkpoint 仅 rank0 保存 |

### 7.9 Definition of Done（强化版）

任务只有同时满足以下全部条件才可标记为完成：

1. 实现已提交，没有通过静默 fallback 掩盖错误。
2. 至少有一条针对原缺陷的回归测试，且 **collected 测试总数不低于当前基线（483）**。
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
| v1.1（已完成） | 稳定性与工程优化：阶段 0-5 全部完成（门禁 + 3 个官方配置 + 20 项缺陷 + 打包/CI + 性能） |
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
| v1.1（已实施） | DINO 去噪目标改用真实 `image_hw`（对应 7.5 的 3-1） | **训练语义变更**：官方配置（3×stride2 骨干）下原代理值是真实尺寸的 **2 倍**，去噪分支与主分支的 `bbox_embed` 被训练在两个坐标系里；修正后需重新训练评估效果。若在 batch 无 `image` 且未配置 `out_stride` 的场景调用，会**直接报错**而不是静默猜尺寸 |
| v1.1（已实施） | epoch 级指标统一以 `val_` 前缀写入 `history.csv`（对应 7.5 的 3-2） | **输出格式变更**：v1.0.0 的裸列 `accuracy/precision/recall/f1` 变为 `val_accuracy/...`，且不再出现「重复调用 `on_validation_epoch_end` 在已 reset 的累加器上算出的全 0」。解析 `history.csv` 的下游脚本需同步改名；`val_loss` / `val_acc` / `train_*` 名称不变 |
| v1.1（已实施） | 梯度累积分母改为「所在窗口的实际长度」（对应 7.5 的 3-3） | **训练语义变更**：v1.0.0 一律除以 `accumulate_grad_batches`，末尾不足一窗时梯度被系统性低估（如 5 batch + accum=2 的尾窗）。修正后与 `accumulate_grad_batches` 配合的最优学习率需重新标定 |
| v1.1（已实施） | resume 增加一致性校验（对应 7.5 的 3-3） | **行为变更**：`load_checkpoint` 现在默认**拒绝**配置哈希 / 任务类型 / 模型类型不匹配的 checkpoint，并拒绝纯 `state_dict`。依赖旧「无校验直接加载」行为的脚本需显式传 `allow_config_mismatch=True`（类型不匹配仍需匹配的配置） |
| v1.1（已实施） | checkpoint 反序列化增加信任策略（对应 7.6 的 4-3） | **安全行为变更**：`load_backbone` 默认 `weights_only=True`（第三方权重按不可信处理），携带非张量状态的完整 checkpoint 需显式 `trusted=True`；`Trainer.load_checkpoint` 因断点续训必须 pickle，默认 `trusted=True` 并在 docstring 写明风险 |
| v1.1（已实施） | `out_indices` 越界不再静默取模（对应 7.7 的 5-4） | **行为变更**：`out_indices=[5]` 配 3 个 stage 从「悄悄变成 [2]」改为抛 `ValueError`。现有官方配置均未越界（E2E 门禁已验证） |
| v1.1（已实施） | `MetricCollection` 拒绝重复指标名（对应 7.7 的 5-4） | **行为变更**：重名成员此前会在 `compute()` 汇总时静默互相覆盖，现在构造与 `add()` 时抛 `ValueError` |
| v1.1（已实施） | `TransformPipeline` 拒绝同流水线内重复归一化（对应 7.7 的 5-4） | **行为变更**：包含两个 `provides_normalization=True` 变换的管线此前会静默归一化两次，现在抛 `ValueError` |
| v1.1（已实施） | 顶层 import 改为惰性（对应 7.7 的 5-4） | **API 语义变更**：`from dvisionix.config import Config` 不再加载 torch（3~4 s → 0.10 s）；组件注册发生在子模块被导入时，使用注册表前需显式 `import dvisionix.models`（或 training / metrics）。顶层便捷导出仍可用 |
| v1.1（已实施） | 合成数据改为可复现（对应 7.7 的 5-1/5-4） | **行为变更**：`tools/train.py` 的合成数据由「每次运行随机」改为按 `(split, i)` 派生固定种子，同一划分多次运行结果一致 |
| v1.1（已实施） | EMA 续训不再重置影子权重（对应 7.7 的 5-1） | **行为变更**：v1.0.0 的 `on_train_begin` 无条件用当前模型权重重建设 shadow，续训等于把 EMA 状态清零；现在会保留恢复出的影子。另外整型 buffer 不再参与滑动平均 |
