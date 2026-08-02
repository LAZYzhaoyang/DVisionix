<div align="center">

# 🔬 DVisionix v1.0.0

**一个模块化、可扩展、配置驱动的 PyTorch 计算机视觉算法库**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

DVisionix 面向个人科研与工程实践，目标是做一个「自己的算法工具箱」：
骨干、颈部、头部、损失、指标、任务、回调全部**注册即用**，可以像搭积木一样自由组合，
覆盖分类、目标检测、语义/实例分割与自监督对比学习。

---

## ✨ 核心特性

### 🏆 统一训练架构
- **Task 组件**：任务逻辑（分类/检测/分割/自监督/线性评估）与训练循环完全解耦，自定义任务继承 `BaseTask` 即可
- **通用 Trainer**：支持任意任务，无需修改训练代码
- **Callback 系统**：ModelCheckpoint / EarlyStopping / EMA / 知识蒸馏 / 进度条，灵活生命周期钩子
- **工程能力**：AMP、梯度累积、DDP 多卡、断点续训（完整状态）、工作目录隔离（默认在代码库外）

### 🧩 配置驱动 + 即插即用
- 统一 `Registry`：模型/层/损失/指标/数据集/变换/任务全部注册，YAML 配置即可构建
- 模型组件化：`backbones → necks → heads → detectors`，共享算子下沉 `layers/`
- **12 个骨干**（CNN/Transformer/timm）、**3 个颈部**、**35 个头部**、**12 个检测器**、**25+ 损失**
- 顶层 API 完整导出：`from dvisionix.models import ConvNeXtBackbone, FCOSDetector, DINOLoss, ...`

### 📊 完整任务支持
- ✅ 图像分类（含度量学习：ArcFace/CosFace/SphereFace/AdaFace/NormFace/CurricularFace/PartialFC/Circle）
- ✅ 目标检测（FCOS / RetinaNet / YOLOv5/7/8/9/10/11 / DETR / RT-DETR / DeformableDETR / DINO-lite / CenterNet）
- ✅ 语义分割（Seg/FCN/DeepLabV3(+)/UNet/SegFormer(+V2/V3)/PSP/UPerNet/BiSeNet/MaskFormer/Mask2Former/SwinUNet）
- ✅ 实例/全景分割（MaskFormer/Mask2Former + PQ/SQ/RQ）
- ✅ 自监督对比学习（SimCLR）+ 线性评估闭环

### 🛠️ 工具链
- 超参搜索 `tools/hparam_search.py`（参数网格/随机采样，逐 trial 独立进程）
- ONNX 导出 `ONNXExporter`（trace/dynamo 后端、精度验证、dict 多输出）
- 结构化日志（console + file + JSONL + TensorBoard）
- 训练增强：warmup 调度器、梯度裁剪、EMA、知识蒸馏（logits + 特征）、torch.compile、channels_last

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

训练产物（日志 / TensorBoard / 检查点 / 最终配置 / history.csv / best_metrics.csv）全部落在
工作目录 `~/dvisionix_runs/<experiment>/<时间戳>-<配置哈希>/`（代码库外，可通过 `--work-dir` 或
环境变量 `DVISIONIX_WORK_DIR` 覆盖）。

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
│   ├── classifiers/     # LinearClassifier（组合器）
│   ├── segmenters/      # SegmentationModel / SwinUNet（组合器）
│   ├── postprocess.py   # NMS / IoU / 共享契约解码器
│   └── toy/             # 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel / DetHead）
├── training/
│   ├── trainer.py       # 统一训练引擎（Task 驱动 / DDP / AMP / 梯度累积 / resume / torch.compile）
│   ├── tasks/           # BaseTask + 7 个内置任务（分类/检测/分割/多标签/SimCLR/线性评估/MaskFormer）
│   ├── callbacks/       # ProgressBar / ModelCheckpoint / EarlyStopping / EMA / DistillCallback
│   ├── optim/           # OPTIMIZERS / SCHEDULERS 注册表（含 warmup 调度器）
│   └── workdir.py / builder.py / checkpoint.py / evaluation.py
├── metrics/             # 分类/分割/检测(mAP)/全景(PQ) + MetricCollection + 预设
├── utils/               # 设备工具 + logging（console/file/JSONL/TensorBoard）
└── export/              # ONNXExporter（trace/dynamo、dict 多输出、精度验证）
tools/
├── train.py             # 配置驱动统一训练入口
└── hparam_search.py     # 超参搜索工具
configs/                 # 各任务合成数据 demo 配置（分类/检测/分割）
```

---

## 📖 文档索引

| 文档 | 说明 |
|------|------|
| [快速开始](docs/quick_start.md) | 安装、Config 驱动训练、编程式训练、训练工程增强 |
| [配置系统](docs/config_system.md) | Config 加载 / 继承 / CLI 覆盖 / schema 校验 |
| [数据模块](docs/data.md) | Sample 协议、BaseDataset、transforms、数据集工具箱 |
| [骨干网络](docs/backbones.md) | 12 种内置骨干与预训练加载 |
| [自定义模型](docs/custom_models.md) | 自定义 Layer / 注册 / 配置驱动组装 |
| [目标检测](docs/detection.md) | 组件化检测器、decode、mAP 评估 |
| [语义分割](docs/segmentation.md) | 分割数据格式与端到端训练 |
| [指标](docs/metrics.md) | 原子指标、组合容器、预设 |
| [日志系统](docs/logging.md) | TrainingLogger / JSONL / TensorBoard |
| [模型导出](docs/model_export.md) | ONNX 导出与验证 |

项目规划与开发约束（model 模块调用规则 R1-R7）见 [CodePlan](CodePlan.md)。

---

## ✅ 测试

```bash
conda run -n dvisionix python -m pytest tests/ -q
```

当前全量测试：**286 passed + 2 skipped**（多卡冒烟测试需 2+ GPU 时自动跳过），ruff / black 全绿。

```bash
ruff check dvisionix tools tests
black --check dvisionix tools tests
```

---

## 📝 版本说明

- **v1.0.0**：功能基线（配置驱动 + 组件化模型库 + 训练工程 + 工具链）确定，API 冻结进入稳定期。
- 历史变更记录已精简并入 [CodePlan](CodePlan.md)。

---

## 📄 许可证

MIT License

## 👤 作者

Zhaoyang Li

---

## 📞 问题反馈

如有问题或建议，请在 GitHub 提交 Issue。
