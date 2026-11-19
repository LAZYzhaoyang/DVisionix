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

- **版本**：v2.0.1（v1.0.0 功能已冻结；v1.1 稳定性与工程修复见已归档的
  [v1.1 计划](docs/history/v1.1_plan.md)）。
- **测试**：v1.0.0 基线为 `286 passed + 2 skipped`、`ruff` / `black` 全绿 —— 已实测复现。
  完整基准数字见第五章「实测数据」。**测试状态一律以 CI 结果为准**，本文件不手写数字。
- **已核实的关键缺口**：19 个官方配置中 3 个无法训练、`decode()` 统一契约实际 2/11 违约、
  EMA 无配置入口、docstring 3 处损坏等，共 20 项，完整清单与证据见归档计划的 **7.1.2**；
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

### ✅ v2（2.0.0）已完成：P0–P5

| 阶段 | 内容 | 关键交付 | 验收证据 |
|---|---|---|---|
| P0 | 架构基线 | 统一异常族、`layers.yaml` + AST 分层检查器、registry 单源 | 197 文件 0 违规 |
| P1 | DVX 标准数据集层 | `data/spec`（schema/classes/split/scan）、folder + labelme 解析、三种非破坏切分、`DVXDataset`、`tools/init_dataset.py` / `check_dataset.py` | 四套真实数据 spec：cls 2210 / det 1646 / multicls 2458 / seg 766 |
| P2 | 训练接线 + 三方契约 + DDP | 三级契约校验、`Task.build_transforms`、`tools/train.py` 接入 DVX、`strategy: auto`、跨卡聚合改 CPU 中转 | cls/det/seg 真实数据各 1 epoch；2 卡 DDP 冒烟与真实训练通过 |
| P3 | 路由式多任务分类 | `MultiTaskClassifier`、`MultiTaskClassificationTask`、字典类别数契约 | multicls 单卡 / 2 卡 1 epoch，per-task 指标落 `history.csv` |
| P4 | 推理链路 | `PreprocessSpec/Pipeline`、`Results`、三个 `Predictor`、`tools/infer.py` | 真实图片 cls/det/seg 推理；检测框回原图坐标、分割 mask 回原尺寸 |
| P5 | 可扩展导出体系 | `BaseExporter` / `EXPORTERS` / `ExportResult` / `VerifyReport`、`tools/export.py`、TensorRT 占位 | 三任务 ONNX 导出 + ORT 一致性 max_diff < 1e-4 |

- **接口契约**（历史归档）：`docs/history/design/v2_p0_p1_contract.md`、
  `docs/history/design/v2_p2_contract.md`、`docs/history/design/v2_p3_contract.md`、
  `docs/history/design/v2_p4_contract.md`、`docs/history/design/v2_p5_contract.md`。
- **使用文档**：`docs/dataset_spec.md`、`training_contracts.md`、`multi_task.md`、`inference.md`、
  `model_export.md`；示例配置在 `configs/examples/`。
- **文档同步（v2.0.0）**：README / CHANGELOG / `docs/quick_start.md` / 上述五篇使用文档 /
  `docs/README.md` 索引均已更新；17 篇 md 链接检查 0 失效；CodePlan 本表与第八章同步登记。
- **文档归档（v2.0.1）**：历史文档已归档到 `docs/history/`（`docs/history/v1.1_changes.md`、
  由原第七章拆出的 `docs/history/v1.1_plan.md`、`docs/history/design/v2_p*_contract.md`），
  `docs/README.md`、README、CHANGELOG 与全库引用同步更新，链接检查 0 失效。
- **当前基线（v2.0.1 清理后）**：`pytest tests -q` → **1089 passed / 1 skipped / 0 failed**；
  `ruff` / `black` 全绿；分层检查器 0 违规。
- **v2.0.1 清理说明**：移除废弃的 `use_giou`（删除 4 条旧开关用例、新增 1 条回归，
  1092 → 1089）；`restore_best_weights` 真正接线；`export_best_onnx` 复用 `build_exporter`
  并写 `best.onnx.json` sidecar；清理未跟踪构建产物。
- **提交**：`deca03b`(P0) / `80d774d`(P1) / `000e9dd`(P2) / `71e38f0`(P3) / `3a282f5`(P4) /
  `2360442`(P5)；发布提交与 `v2.0.0` tag 见第八章。

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

> **v2.0.0 基线（2026-11-18）**：`pytest tests -q` → **1092 passed / 1 skipped / 0 failed**
> （Python 3.11 / torch 2.14.1+cu130 / 7×RTX 5060 Ti，2 卡可见）；`ruff` / `black` 全绿；
> 分层检查器 0 违规。上表为 v1.0.0 / v1.1 历史基准，保留用于对照。
>
> **v2.0.1 基线（2026-11-20）**：清理后 `pytest tests -q` →
> **1089 passed / 1 skipped / 0 failed**（删除 4 条 `use_giou` 旧开关用例、新增 1 条
> `restore_best_weights` 回归，1092 → 1089）；`ruff` / `black` / 分层检查器全绿。

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

### 🔄 历史状态：v1.1 稳定性与工程优化（已完成）

- **第七章已整体归档到 [docs/history/v1.1_plan.md](docs/history/v1.1_plan.md)**；
  当前计划见本章「v2 剩余计划」，行为变更登记见第八章。
- **阶段 0-5 已全部完成**（门禁 → 配置修复 → P0 正确性 → 训练/评估/导出闭环 →
  打包与 CI → 性能优化），见归档计划的 7.2 / 7.3 / 7.4 / 7.5 / 7.6 / 7.7 完成标记。
  20 项已核实缺陷 D1-D20 全部修复并附回归测试；官方配置由 14/17 可跑提升到 **17/17 全绿**。
- **v1.1 的开发工作到此结束**，可以进入第六章的 P1/P2/P3（模型库继续扩充等），
  或按需把 mypy / pip-audit 从信息性门禁升级为阻断（见归档计划的 7.6 升级条件）。
- **v1.1 之后另有一批 v1.1.1 修复**（正确性与稳定性，含会改变检测指标口径的解码修复），
  记录见归档计划的 **7.10**，行为变更登记见第八章。该批次完成后 collected 测试总数由 483 增至 **531**。
- **v1.1 早期一批未提交改动已整体回退**（回退原因与更正见归档计划的 7.1.1，
  补丁留档于 `.dev_archive/wip-v1.1-partial.patch`，已在 `.gitignore` 中排除）。
- 该批次中**诊断正确且实现无误**的两项（DDP 递归聚合、FCOS/YOLO 重复累加 L1）
  已在阶段 2 重新实现，并补齐其缺失的归一化与测试。
- **用户文档**：新增 [docs/history/v1.1_changes.md](docs/history/v1.1_changes.md)（迁移指南，v2.0.1 起归档），
  并为 7 篇专题文档补充「v1.1 变更要点」章节、更新文档索引与 README。

### ⏸️ 环境阻塞

- 多卡 NCCL 验证（原第六章 P0）：当前环境为**单张** CUDA 卡（RTX 3060），无法执行；
  替代方案（CPU gloo 双进程 + 单卡 CUDA + DDP 可控单元测试）已落地，解除条件见归档计划的 **7.8**。
- 受限环境下 `test_ddp_cpu.py` 的 2 条多进程用例会失败（沙箱禁止命名管道），
  非项目缺陷，见归档计划的 **7.8**。

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

### v2 剩余计划（2.0.0 之后，按优先级）

1. **B. 检索训练**（优先）：`EmbeddingModel` + `SupConLoss` + PK 采样 + `Recall@K` / `mAP@K`；
   数据侧复用 DVX 分类布局，指标侧新增检索指标族。
2. **C. 公开数据集 adapter**：COCO / VOC / Cityscapes / ADE20K → DVX 格式 converter，
   复用 `tools/init_dataset.py` 的体检与切分能力。
3. **D. ONNX / TensorRT 运行时推理**：ORT Session 封装 + TensorRT 后端（当前仅导出，
   `TensorRTExporter.export` 为占位）。

### P1 — 指标 torchmetrics 迁移（low）
- **背景**：内置指标（mAP / mIoU / PQ）正确性已由测试保障，torchmetrics 作为可选后端。
- **实现步骤**：① `metrics/detection.py` 增加 `backend="torchmetrics"` 分支；② 分割/全景指标同法；
  ③ 保持 `BaseMetric` 接口不变（update/compute/reset），配置 `metrics.backend` 切换。
- **验收**：内置与 torchmetrics 后端在同一合成数据上结果一致（容差 1e-4）。

### P2 — 模型库继续扩充（medium）
- 骨干：更多 Transformer 变体（ConvNeXtV3 趋势、MoE 视觉骨干等，按最新论文跟进）。
- 检测：YOLO 系列完整化（YOLOv12 等）、DETR 家族（Co-DETR / DINOv2 集成）。
- 分割：Mask2Former 完整训练增强、SAM 类交互分割（另行立项）。
- 新增组件遵循 R1-R7，注册即用，配套合成数据配置与测试，**并纳入配置 E2E 门禁**（原 7.2）。

### P3 — 训练工程增强（low）
- 自动混合精度扩展（bf16）、DeepSpeed/FSDP 大模型训练（按需）。
- 实验管理 Web 化（可选：接入 wandb/MLflow 日志后端）。

### ⏸️ 原「P0 — 多卡实验验证」→ 已移至归档计划的 7.8（环境阻塞）

---

## 七、v1.1 稳定性与工程优化计划（已归档）

> 本章完整内容已归档到 [docs/history/v1.1_plan.md](docs/history/v1.1_plan.md)：
> 执行原则（7.0）、v1.0.0 论断核实表（7.1）、阶段 0-5 完成记录（7.2-7.7）、
> 环境阻塞（7.8）、Definition of Done（7.9）与 v1.1.1 修复批次（7.10）。
>
> **摘要**：v1.1 按「门禁先行 → 修复官方配置 → P0 正确性 → 训练/评估/导出闭环 →
> 打包与 CI → 性能」六个阶段推进，修复 20 项已核实缺陷（D1-D20），官方配置由
> 14/17 可跑提升到 **17/17 全绿**；v1.1.1 追加 11 项正确性与稳定性修复
> （N-1..N-11），其中检测解码索引修正会改变检测指标口径。
>
> v1.1 已完结，本章属历史记录，**不是当前执行计划**；当前规划见第五、六章，
> 行为变更登记见第八章。

---

## 八、版本记录
### 版本里程碑（精简）

| 版本 | 里程碑 |
|---|---|
| v2.0.1（当前） | **清理与修复**：移除废弃的 `use_giou`；`training.early_stopping.restore_best_weights` 真正接线；`training.export_best_onnx` 复用 P5 `build_exporter` 并写 `best.onnx.json` sidecar；清理未跟踪构建产物；历史文档归档到 `docs/history/`。测试 1089 passed / 1 skipped / 0 failed |
| v2.0.0 | **v2 重构发布**：DVX 标准数据集层、训练接线与任务/模型/数据契约、路由式多任务分类、推理链路、可扩展导出体系（含 `ONNXExporter.export/verify` 返回类型变更与契约校验）。测试 1092 passed / 1 skipped / 0 failed |
| v1.1.1 | 正确性与稳定性修复：模型层（解码索引、AMP 数值、损失语义）、GPU 可用性、训练可靠性、DDP 正确性、工具链与文档收口。**无新增功能**，故按补丁号发布；含会改变检测指标口径的修复（见归档的 7.10 与本章行为变更登记） |
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
> 表中「7.x」编号指已归档的 [docs/history/v1.1_plan.md](docs/history/v1.1_plan.md)。

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
| v2.0.0 | `ONNXExporter.export()` 返回 `ExportResult`、`verify()` 返回 `VerifyReport`（P5） | **API 返回类型变更**：`ExportResult` 实现 `__str__`/`__fspath__`、`VerifyReport` 实现 `__bool__`，既有的「拿路径」与「真值判断」用法保持可用；用 `== str` 直接比较返回值或 `verify() is True` 的代码需调整 |
| v2.0.0 | 训练入口执行任务 / 模型 / 数据三级契约校验（P2/P3） | **行为变更**：任务与模型能力无交集、DVX `data_type` 不匹配、`num_classes` 不一致、batch 缺字段，都会在训练开始前抛领域异常；未声明能力的自定义任务跳过对应校验 |
| v2.0.0 | `ClsHead` / `MultiLabelHead.forward` 接受 `labels`（P2） | **训练语义变更**：`LinearClassifier + cls_head + ClassificationTask` 此前第一个 batch 即 `TypeError`；修复后 margin 头在分类任务中真正生效，相关超参需重新标定 |
| v2.0.0 | `strategy: auto` 识别 torchrun；验证指标聚合先转 CPU（P2） | **行为变更**：torchrun 下不再需要手动指定 `--strategy ddp`；多卡验证聚合不再因跨设备 `torch.cat` 崩溃 |
| v2.0.0 | `BaseModel.task_type` 类属性声明不再被 `BaseModel.__init__` 的 `None` 覆盖（P2） | **行为变更**：`class MyModel(BaseModel): task_type = "detection"` 现在能被 `model_task_types` 正确读取；此前会被实例属性覆盖为空 |
| v2.0.1 | 移除 `use_giou` 参数（`FCOSDetectionLoss` / `YOLOLoss`），只保留显式 `giou_weight` / `l1_weight` | **配置 / API 变更**：v1.1 起已发 `DeprecationWarning` 的旧开关彻底删除，配置或代码里再写 `use_giou` 会在构建损失时抛 `TypeError`（未知参数）。迁移方式：`use_giou: true` → 保持默认（GIoU only），`use_giou: false` → `giou_weight: 0.0, l1_weight: 1.0`；仓库内已零使用 |
| v2.0.1 | `training.early_stopping.restore_best_weights` 真正生效（此前 `build_callbacks` 未转发，写 `false` 也静默按 `True` 执行） | **训练行为变更**：`false` 时早停触发后**不再恢复最优权重**，模型停留在最后一轮；依赖「早停后拿到 best 权重」的配置若显式写了 `false`，需复核最终产物。默认值仍为 `true`，未显式配置者行为不变 |
| v2.0.1 | `training.export_best_onnx` 改用 P5 `build_exporter("onnx", ...)`，并额外落盘 `best.onnx.json` sidecar | **输出新增**：训练结束除 `work_dir/best.onnx` 外还写 `best.onnx.json`（`ExportResult` 序列化：预处理规格 / 输入输出 shape / metadata）；返回路径不变，失败仍只告警、不影响训练 |
| v1.1.1 | **检测解码的「类别 / 位置」索引修正**：`yolo_decode` / `fcos_decode` / `retinanet_decode` 由 `//n_loc`、`%n_loc` 改为 `%num_classes`、`//num_classes`（详见 7.10 的 N-1） | **指标口径变更（严重）**：展平索引是 `idx = 位置 * 类别数 + 类别`，旧实现把两者互换，导致**标签与框同时错位**（受控输入应得 `label=1、中心(3,3)`，实得 `label=2、中心(1,3)`）。修复前所有检测器的 `val_mAP` / `val_mAP_50` / `val_mAP_75`、推理输出与 ONNX 导出结果**均不可信**；训练 loss 因损失端语义正确而始终正常，故此前无任何信号。**历史检测指标不可与修复后比较** |
| v1.1.1 | `RetinaNetLoss` 补上 batch / 层数归一化，与 FCOS / YOLO 口径一致（7.10 的 N-2） | **训练语义变更**：旧实现是逐项均值的累加，loss 与 `batch_size × 层数` 成正比（实测 bs=1/2/4 → 0.971 / 1.942 / 3.884），等效学习率随 batch 漂移。修正后 loss 量级显著下降，**RetinaNet 的既有学习率需重新标定**，且修复前后的 `val_loss` 不可比 |
| v1.1.1 | ArcFace / CosFace / SphereFace / AdaFace / CurricularFace 的 margin **开始真正生效**：`LinearClassifier` 透传 `**kwargs`，`ClassificationTask.training_step` 传入 `labels`（7.10 的 N-3） | **训练语义变更**：此前 kwargs 被丢弃且任务只传 `images`，5 个 margin 头**从未施加过 margin**，等价于 NormFace。修正后这些头的 logits 分布与最优超参都会变化，需重新训练与标定 |
| v1.1.1 | `SegFormerV3Head` 的通道注意力由 `fused * se(fused)` 改为 `se(fused)`（7.10 的 N-4） | **训练语义变更**：`SEBlock.forward` 已返回 `x * scale`，旧写法实为 `fused² · scale`（符号信息被抹掉，`|fused|>256` 时 fp16 溢出）。输出分布改变，需重新训练 |
| v1.1.1 | `UPerNetHead.fpn_convs` 真正参与前向（7.10 的 N-4） | **训练语义变更**：该模块此前是**死参数**（定义后从未使用），既永远拿不到梯度，也会在 DDP 下因 `find_unused_parameters=False` 于第二轮中断训练。现在融合后增加一次 3×3 平滑（UPerNet 原设计），输出与参数量均有变化 |
| v1.1.1 | `DiceLoss` / `CombinedSegmentationLoss` 正确排除 `ignore_index` 与越界标签（7.10 的 N-5）；`MaskFormerLoss` 的 mask BCE 改 `binary_cross_entropy_with_logits` | **行为变更**：v1.1 下 `255` 会原样进入 `F.one_hot` 并抛 `RuntimeError`，即**含 void 标签的真实数据集（Cityscapes / ADE20K）完全不可用**；现在可正常计算（被忽略像素的 one-hot 与概率均清零）。BCE 改动使其可在 `autocast(fp16)` 下运行，数值上与 `sigmoid`+BCE 等价 |
| v1.1.1 | 非标量指标（`per_class=True` / `average="none"`）在 epoch 汇总前展开为 `key_0 / key_1 / ...`（7.10 的 N-6） | **输出格式变更**：旧行为是 `float(list)` 抛 `TypeError`，训练在**第一个 epoch 末尾**崩溃。现在 `history.csv` / TensorBoard 中出现 `val_dice_0` 一类子键；标量与张量原样透传，不受影响 |
| v1.1.1 | checkpoint 的 `model_state_dict` 统一保存**未包装**的键，加载时按目标模型自动剥离/补回 `module.` 与 `_orig_mod.` 前缀（7.10 的 N-7） | **兼容性变更**：旧 checkpoint（含包装前缀）仍可加载（自动剥离），但旧版本代码读新 checkpoint 时若模型被包装会需要前缀匹配。修复前 `load_backbone` 遇包装前缀会**静默一个权重都不加载**，EMA 的 key 过滤同样静默失效 |
| v1.1.1 | DDP 下 `history.csv` / `best_metrics.csv` / JSONL / TensorBoard 仅由 rank0 写出；早停只在 rank0 决策并在 epoch 末广播（7.10 的 N-8） | **行为变更**：此前所有 rank 并发写同一路径，且 rank≠0 的 `val_*` 恒为 0，CSV 中可能出现**整列 0 的假指标**；早停各 rank 独立决策会导致 rank0 先退出后其余 rank 永久 hang |
| v1.1.1 | `torchrun` 下工作目录改用确定性的 `<配置哈希>` 命名（不带时间戳）（7.10 的 N-8） | **输出路径变更**：此前目录名含各进程独立取的 `datetime.now()`，跨秒启动就会让 rank0 的 checkpoint/日志与其它 rank 的 CSV **分家**，`resume auto` 也会选错。单进程行为不变（仍带时间戳） |
| v1.1.1 | `--devices` / `training.devices` 真正生效；显式 `device: "cpu"` 不再被 DDP 覆盖；NCCL/Gloo 按实际设备选择（7.10 的 N-9） | **行为变更**：`devices` 此前是死参数，`--devices 1,3` 会被静默忽略并改用 0、1 号卡。现在按 `devices[local_rank]` 绑定并显式 `set_device`；在 GPU 机器上做 CPU 回归时不会再被强制绑卡 |
| v1.1.1 | `resume: true` 归一为 `"auto"`（7.10 的 N-10） | **行为变更**：此前 `true` 被原样当作路径传给 `torch.load`（抛 `AttributeError: 'bool' object has no attribute 'seek'`），且会另起一个带新时间戳的目录。注意 `resume` 参与配置哈希，**在配置里新增**该键会被续训一致性校验拒绝；推荐用命令行 `--resume auto` |
| v1.1.1 | `ONNXExporter.export(backend="dynamo")` 默认参数即可用；`dynamic_batch` / `dynamic_size` 默认改为 `None` 哨兵（7.10 的 N-11） | **API 行为变更**：`dynamic_batch` 原默认 `True`，与"dynamo 不支持 `dynamic_axes`"的硬校验冲突，使最自然的默认调用**必然抛 ValueError**。现在 `trace` 仍默认 `True`（历史行为不变），`dynamo` 默认 `False`；显式传 `dynamic_batch=True` 时仍报错 |
