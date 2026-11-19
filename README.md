<div align="center">

# 🔬 DVisionix v2.0.1

**一个模块化、可扩展、配置驱动的 PyTorch 计算机视觉算法库**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

DVisionix 面向个人科研与工程实践，目标是做一个「自己的算法工具箱」：
骨干、颈部、头部、损失、指标、任务、回调全部**注册即用**，可以像搭积木一样自由组合，
覆盖分类、目标检测、语义/实例分割、路由式多任务分类与自监督对比学习。

---

## ✨ 核心特性

### 🏆 统一训练架构
- **Task 组件**：任务逻辑（分类/检测/分割/多任务/自监督/线性评估）与训练循环完全解耦，自定义任务继承 `BaseTask` 即可
- **通用 Trainer**：支持任意任务，无需修改训练代码
- **Callback 系统**：ModelCheckpoint / EarlyStopping / EMA / 知识蒸馏 / 进度条，灵活生命周期钩子
- **训练契约校验**：任务↔模型能力、任务↔DVX `data_type`、batch 必需字段三级校验，
  错配在训练开始前给出可操作的修复提示
- **自适应 DDP（`strategy: auto`）**：torchrun 下自动识别为 DDP 并跨卡聚合验证指标；
  `device: auto` 自动选择设备
- **工程能力**：AMP、梯度累积、DDP 多卡、断点续训（完整状态）、工作目录隔离（默认在代码库外）

### 🧩 配置驱动 + 即插即用
- 统一 `Registry`：模型/层/损失/指标/数据集/变换/任务全部注册，YAML 配置即可构建
- 模型组件化：`backbones → necks → heads → detectors`，共享算子下沉 `layers/`
- **12 个骨干**（CNN/Transformer/timm）、**3 个颈部**、**35 个头部**、**12 个检测器**、**25+ 损失**
- 顶层 API 完整导出：`from dvisionix.models import ConvNeXtBackbone, FCOSDetector, DINOLoss, ...`

### 📊 完整任务支持
- ✅ 图像分类（含度量学习：ArcFace/CosFace/SphereFace/AdaFace/NormFace/CurricularFace/PartialFC/Circle）
- ✅ 路由式多任务分类（共享骨干只前向一次，按样本 `task` 路由到各任务分类头，各任务标签空间独立）
- ✅ 目标检测（FCOS / RetinaNet / YOLOv5/7/8/9/10/11 / DETR / RT-DETR / DeformableDETR / DINO-lite / CenterNet）
- ✅ 语义分割（Seg/FCN/DeepLabV3(+)/UNet/SegFormer(+V2/V3)/PSP/UPerNet/BiSeNet/MaskFormer/Mask2Former/SwinUNet）
- ✅ 实例/全景分割（MaskFormer/Mask2Former + PQ/SQ/RQ）
- ✅ 自监督对比学习（SimCLR）+ 线性评估闭环

### 🔎 推理链路（v2 P4）
- **预处理规格**：`PreprocessSpec` / `PreprocessMeta` 记录尺寸与缩放比；
  复用验证期 `*Transforms(train=False)`，与训练完全一致
- **统一 Results**：分类 `probs/topk`、检测 `boxes/scores/labels`、分割 `mask`，
  几何量已反变换回**原图坐标 / 原尺寸**
- **三任务 Predictor**：`build_predictor` / `BasePredictor.from_checkpoint`，
  支持 classification / detection / segmentation
- **CLI**：`tools/infer.py` 配置驱动，单图/目录批量输出 JSON 与可视化；
  详见 [推理](docs/inference.md)

### 📤 模型导出（v2 P5）
- **可扩展导出体系**：`BaseExporter` / `EXPORTERS` 注册表 / `ExportResult`
  （含 `PreprocessSpec` 预处理规格），`build_exporter("onnx" | "tensorrt", ...)` 按名构建
- **ONNX 一致性校验**：`ExportResult` + `VerifyReport`（全局与逐输出 `max_abs_diff`，
  `bool(report)` 保持旧布尔语义），trace / dynamo 双后端、嵌套输出、动态 batch
- **CLI**：`tools/export.py` 配置驱动导出 + `--verify`，落盘 `<output>.json` sidecar；
  TensorRT 仅预留扩展点（`export()` 抛 `NotImplementedError`），详见
  [模型导出](docs/model_export.md)

### 📦 标准化数据
- **DVX 标准数据集**：一个根目录 + 一份 `dvx_dataset.yaml`，覆盖**分类 / labelme 检测与分割 / 多任务分类**；
  单任务共用一张类别表（多任务每任务独立类表），切分非破坏性（ratio / dir / list），
  `DVXDataset` 直接加载。
  配套 `tools/init_dataset.py`（生成草稿）与 `tools/check_dataset.py`（只读体检），
  详见 [DVX 标准数据集](docs/dataset_spec.md)。

### 🛠️ 工具链
- 超参搜索 `tools/hparam_search.py`（参数网格/随机采样，逐 trial 独立进程）
- 性能基准 `tools/benchmark.py`（固定输入、可复现、可隔离测峰值内存）
- 模型导出 `tools/export.py`（ONNX 一致性校验，导出体系见「模型导出」）
- 结构化日志（console + file + JSONL + TensorBoard）
- 训练增强：warmup 调度器、梯度裁剪、EMA（`training.ema` 配置开启）、
  知识蒸馏（编程式 API：`DistillCallback`）、torch.compile、channels_last

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/LAZYzhaoyang/DVisionix.git
cd DVisionix

# 创建 conda 环境（或使用已有 Python 3.10+ 环境）
conda create -n dvisionix python=3.10 -y
conda activate dvisionix

# 安装 PyTorch（按你的 CUDA 版本选择，CPU 可省略 cuda 参数）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装 DVisionix 本体 + 完整/开发依赖
pip install -e .[dev,full]
```

依赖与版本元数据的**唯一来源是 `pyproject.toml`**（`[project]` 与
`[project.optional-dependencies]`）；`requirements.txt` 只是指向它的指针，不再单独声明版本。
可选能力按需安装：

```bash
pip install -e .              # 仅核心依赖
pip install -e .[export]      # ONNX 导出与验证
pip install -e .[models]      # 第三方骨干（timm）
pip install -e .[datasets]    # 数据增强（albumentations）
pip install -e .[coco]        # COCO 评估（pycocotools）
pip install -e .[metrics]     # 可选指标后端（torchmetrics）
pip install -e .[full,dev]    # 全部能力 + 开发工具
```

**支持的解释器**：`requires-python = ">=3.10"`，CI 覆盖 3.10 / 3.11 / 3.12
（classifier 已列到 3.14；3.13 / 3.14 尚未纳入 CI 矩阵，属实验性支持）。

### 30 秒上手：Config 驱动训练（合成数据）

```bash
# 分类：合成数据快速验证
python tools/train.py --config configs/classification/demo_synthetic.yaml

# 覆盖单个参数
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --cfg-options training.num_epochs=5 training.optimizer.lr=0.01

# 自动续训最近一次任务
python tools/train.py --config configs/classification/demo_synthetic.yaml --resume auto

# 多卡训练（需 2+ 张 GPU）
torchrun --nproc_per_node=2 tools/train.py --config configs/classification/demo_synthetic.yaml --devices 0,1
```

训练产物（日志 / TensorBoard / 检查点 / 最终配置 / `history.csv` / `best_metrics.csv`）全部落在
工作目录 `~/dvisionix_runs/<experiment>/<时间戳>-<配置哈希>/`（代码库外，可通过 `--work-dir` 或
环境变量 `DVISIONIX_WORK_DIR` 覆盖）。

> `--work-dir` 指定的是**工作根**：实际目录是在其下再拼 `<experiment>/<时间戳>-<配置哈希>`。
> 通过 `torchrun` 启动时，目录名改为确定性的 `<配置哈希>`（不带时间戳），
> 以保证所有 rank 落在同一目录；DDP 下 `history.csv` / `best_metrics.csv` / TensorBoard
> 仅由 rank0 写出（rank≠0 的 `val_*` 恒为 0，写出来是假指标）。

### 60 秒上手：用自己的数据（DVX 标准数据集）

```bash
# 1) 扫描数据目录，生成规格草稿（只读，不移动/改名任何原始文件）
python tools/init_dataset.py --root ./data/my_cls --data-type classification \
  --out ./data/my_cls/dvx_dataset.yaml

# 2) 人工确认 classes 顺序后体检，并预览 train/val 切分
python tools/check_dataset.py --spec ./data/my_cls/dvx_dataset.yaml --split-preview

# 3) 用 DVXDataset 加载（分类示例）
python -c "from dvisionix.data import DVXDataset; ds = DVXDataset('./data/my_cls/dvx_dataset.yaml', split='train'); print(len(ds), ds[0]['label'])"

# 4) 用 DVX 数据集训练（分类示例；检测/分割/多任务见 configs/examples/）
python tools/train.py --config configs/examples/dvx_classification.yaml \
  --cfg-options data.spec=./data/my_cls/dvx_dataset.yaml --work-dir /tmp/dvx_cls
```

`DVXDataset` 支持分类、labelme 检测/分割、多任务分类；字段说明、三种切分模式与四类真实数据
见 [DVX 标准数据集](docs/dataset_spec.md)，数据模板在 `configs/datasets/`
（[分类](configs/datasets/classification_folder.example.yaml) 等四份）。
训练入口通过 `data.dataset: dvx` + `data.spec` 构建它；单任务训练的 `model.num_classes`
必须与 spec 类别数一致，由契约校验兜底。检测/分割配置模板见 `configs/examples/`，
路由式多任务分类（`<任务>/<类别>/*.jpg`，每任务一个分类头）见
[多任务分类](docs/multi_task.md)，三级校验与自适应训练见
[训练契约与自适应训练](docs/training_contracts.md)。

### 用训练好的模型推理

```bash
# 完整 Trainer checkpoint 含非张量状态，需要 --trusted
python tools/infer.py --config configs/classification/demo_synthetic.yaml \
  --checkpoint /path/to/work_dir/checkpoints/best.pt \
  --input ./samples --output ./outputs/cls --trusted --save-vis
```

每张图输出 `<stem>.json`，`--save-vis` 时另存 `<stem>_vis.jpg`；检测/分割把
`--config` 换成对应配置即可。预处理规格、`Results` 字段、反变换与已知限制见
[推理链路](docs/inference.md)。

### 导出为 ONNX

```bash
python tools/export.py --config configs/classification/demo_synthetic.yaml \
  --checkpoint /path/to/work_dir/checkpoints/best.pt \
  --output ./exports/cls.onnx --verify --trusted
```

导出同时在 `<output>.json` 写入 sidecar；`--verify` 通过时打印
「一致性校验: 通过，全局 max_abs_diff=...」，未通过退出码 1。完整参数表、`ExportResult`
字段与 TensorRT 扩展点见 [模型导出](docs/model_export.md)。

### 编程式使用

```python
from dvisionix.config import Config
from dvisionix.models import build_model
from dvisionix.training import Trainer, build_task

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())
task = build_task({"type": "ClassificationTask", "num_classes": 10})
trainer = Trainer(task=task, train_loader=train_loader, val_loader=val_loader,
                  max_epochs=10, amp=True, seed=42)
trainer.fit(model)
```

---

## 🏗️ 架构总览（v1.0.0）

```
dvisionix/
├── registry.py          # 统一注册表（MODELS/BACKBONES/NECKS/HEADS/LOSSES/METRICS/TASKS/...）
├── config/              # Config：YAML 继承 / CLI 覆盖 / schema 校验（defaults/ 内置各任务默认配置）
├── data/                # Sample 协议 + BaseDataset + 原子变换 + 公开数据集工具箱 + 自定义数据集
├── models/
│   ├── base.py          # BaseModel（任务类型校验 / init_weights / count_parameters）
│   ├── layers/          # 通用算子：ConvNormAct/CSP/ELAN/可变形注意力/窗口注意力/PatchOps/anchors 等
│   ├── backbones/       # 12 种骨干（Sequential/ConvNeXt(+V2)/CSPDarknet/MobileNetV3/EfficientNetLite/ViT/Swin(+V2)/MiT/timm）
│   ├── necks/           # FPN / PANet / PixelDecoder
│   ├── heads/           # 分类(11) · 分割(14) · 检测(10)，每头一文件
│   ├── losses/          # BaseLoss + LossComposer + 分类/分割/检测损失（含 assigner/matcher）
│   ├── detectors/       # SingleStageDetector 脚手架 + 12 个检测器（decode 与模型同文件）
│   ├── classifiers/     # LinearClassifier（组合器）/ MultiTaskClassifier（路由式多任务）
│   ├── segmenters/      # SegmentationModel / SwinUNet（组合器）
│   ├── postprocess.py   # NMS / IoU / 共享契约解码器
│   └── toy/             # 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel / DetHead）
├── inference/           # 推理：预处理规格 / 统一 Results / 三任务 Predictor
├── training/
│   ├── trainer.py       # 统一训练引擎（Task 驱动 / DDP / AMP / 梯度累积 / resume / torch.compile）
│   ├── tasks/           # BaseTask + 8 个内置任务（分类/检测/分割/多标签/多任务/SimCLR/线性评估/MaskFormer）
│   ├── callbacks/       # ProgressBar / ModelCheckpoint / EarlyStopping / EMA / DistillCallback
│   ├── optim/           # OPTIMIZERS / SCHEDULERS 注册表（含 warmup 调度器）
│   └── workdir.py / builder.py / checkpoint.py / evaluation.py
├── metrics/             # 分类/分割/检测(mAP)/全景(PQ) + MetricCollection + 预设
├── utils/               # 设备工具 + logging（console/file/JSONL/TensorBoard）
└── export/              # 导出体系（BaseExporter / EXPORTERS / ExportResult；ONNX + TensorRT 占位）
tools/
├── train.py             # 配置驱动统一训练入口
├── infer.py             # 配置驱动推理入口（分类/检测/分割）
├── export.py            # 配置驱动导出入口（ONNX + 一致性校验）
└── hparam_search.py     # 超参搜索工具
configs/                 # 各任务合成数据 demo 配置（分类/检测/分割）
```

---

## 📖 文档索引

| 文档 | 说明 |
|------|------|
| [快速开始](docs/quick_start.md) | 安装、Config 驱动训练、编程式训练、训练工程增强 |
| [推理链路](docs/inference.md) | 预处理规格、统一 Results、三任务 Predictor、CLI 与已知限制 |
| [配置系统](docs/config_system.md) | Config 加载 / 继承 / CLI 覆盖 / schema 校验 |
| [数据模块](docs/data.md) | Sample 协议、BaseDataset、transforms、数据集工具箱 |
| [DVX 标准数据集](docs/dataset_spec.md) | dvx_dataset.yaml 字段、三种切分、DVXDataset 用法与四套真实数据 |
| [训练契约与自适应训练](docs/training_contracts.md) | 任务/模型/数据三级契约、8 个内置任务、自定义能力声明、`strategy: auto` / `device` / AMP |
| [路由式多任务分类](docs/multi_task.md) | 共享骨干 + 每任务一个分类头、数据与配置、单卡/DDP 训练、指标命名与限制 |
| [架构与分层规则](docs/architecture.md) | R1-R3、layers.yaml、分层检查器与例外登记流程 |
| [骨干网络](docs/backbones.md) | 12 种内置骨干与预训练加载 |
| [自定义模型](docs/custom_models.md) | 自定义 Layer / 注册 / 配置驱动组装 |
| [目标检测](docs/detection.md) | 组件化检测器、decode、mAP 评估 |
| [语义分割](docs/segmentation.md) | 分割数据格式与端到端训练 |
| [指标](docs/metrics.md) | 原子指标、组合容器、预设 |
| [日志系统](docs/logging.md) | TrainingLogger / JSONL / TensorBoard |
| [模型导出](docs/model_export.md) | 可扩展导出体系（BaseExporter/EXPORTERS/ExportResult）、CLI 与 ONNX 一致性校验 |
| [历史归档](docs/history/README.md) | v1.1 迁移指南、v1.1 计划与 v2 实现契约（历史记录，非当前使用文档） |

项目规划与开发约束（model 模块调用规则 R1-R7）见 [CodePlan](CodePlan.md)。

---

## ✅ 测试

```bash
conda run -n dvisionix python -m pytest tests/ -q
```

测试状态**以 CI 运行结果为准**。当前实测 **1089 passed / 1 skipped / 0 failed**
（v2.0.0 为 1092；v1.1.1 基线为 528 passed / 1 skipped），`ruff` / `black` 全绿。
`test_ddp_cpu.py` 的两条多进程用例需要创建命名管道，在禁止命名管道的受限沙箱中可能失败；
本机与常规 CI 环境可正常运行。
完整对照基准与门禁定义见 [CodePlan](CodePlan.md) 第五章与 [v1.1 计划归档](docs/history/v1.1_plan.md) 的 7.2 节。

除常规测试外，另有几道装配层门禁：

```bash
# 官方配置端到端：configs/ 下每个可训练配置各跑 1 个 epoch
python -m pytest tests/test_e2e_configs.py -q

# 模型契约：所有检测器的 decode() 必须满足统一关键字参数契约
python -m pytest tests/test_model_contracts.py -q

# CPU 双进程 DDP 一致性（无需 GPU）
python -m pytest tests/test_training/test_ddp_cpu.py -q

# wheel 内容与「装完能用」
python -m pytest tests/test_packaging.py -q

# 只要快速反馈时，跳过配置端到端门禁
python -m pytest tests -q -m "not slow"
```

> Windows 提示：请先 `conda activate dvisionix` 再运行测试。直接用解释器绝对路径时
> `Library\bin` 不在 `PATH`，子进程 `import torch` 会报 `DLL load failed`。

```bash
ruff check dvisionix tools tests
black --check dvisionix tools tests
mypy                                  # 当前基线 121 errors / 39 files，CI 中为非阻断
```

CI 门禁顺序：`ruff` → `black` → `mypy`（信息性）→ 快速测试 → 全量测试 + 覆盖率（≥90%）
→ wheel 构建与干净环境安装 → `pip-audit`（信息性）。

---

## 📝 版本说明

- **v2.0.1（当前）**：清理与修复 —— 移除废弃的 `use_giou`（统一用
  `giou_weight` / `l1_weight`）；`training.early_stopping.restore_best_weights`
  真正生效（此前写了也静默无效）；`training.export_best_onnx` 复用导出体系并额外落盘
  `best.onnx.json`；历史文档归档到 `docs/history/`。详见
  [CHANGELOG.md](CHANGELOG.md) 的 2.0.1 章节。
- **v2.0.0（上一版）**：v2 重构发布 —— DVX 标准数据集层、训练接线与任务/模型/数据契约、
  路由式多任务分类、推理链路（预处理规格 / Results / Predictor / CLI）、
  可扩展导出体系（BaseExporter / EXPORTERS / ONNX 一致性校验）。
  ⚠️ **含接口与行为变更**（`ONNXExporter.export/verify` 返回类型、契约校验拦截错配、
  `ClsHead` 开始接受 `labels` 等），升级前请读
  [CHANGELOG.md](CHANGELOG.md) 的 2.0.0 章节。
- **v1.1.1**：正确性与稳定性修复（模型层 / GPU 可用性 / 训练可靠性 / DDP / 工具链），
  ⚠️ **其中包含会改变检测指标口径的修复**（解码索引错位、RetinaNet 损失归一化、
  margin 头开始真正生效等），历史实验指标不可直接比较。
  完整清单见 [CHANGELOG.md](CHANGELOG.md) 的 1.1.1 章节。
- **v1.1.0**：稳定性与工程优化。修复 20 项已核实缺陷，官方配置由 14/17 可跑
  提升到 17/17；新增装配层门禁与性能基准；打包与 CI 收口。
  **升级前请先读 [v1.1 变更与迁移指南](docs/history/v1.1_changes.md)** ——
  其中有会改变训练结果与输出格式的变更。
- **v1.0.0**：功能基线（配置驱动 + 组件化模型库 + 训练工程 + 工具链）确定，API 冻结进入稳定期。
- 完整版本历史见 [CHANGELOG.md](CHANGELOG.md)；规划与开发约束（R1-R7）见 [CodePlan](CodePlan.md)。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

## 👤 作者

Zhaoyang Li

---

## 📞 问题反馈

如有问题或建议，请在 GitHub 提交 Issue。
