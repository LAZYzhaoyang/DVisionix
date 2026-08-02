<div align="center">

# 🔬 DVisionix

**一个模块化、可扩展的深度学习算法库

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ✨ 核心特性

### 🏆 真正通用的训练架构
- **Task 系统**：任务逻辑与训练循环完全解耦
- **Callback 系统**：灵活的生命周期钩子
- **通用 Trainer**：支持任意任务，不需要修改训练代码

### 📦 统一的数据接口
- 所有任务（分类、检测、分割）使用相同的 BaseDataset 基类
- 统一的字典格式输出
- build_dataset 配置驱动构建数据集（注册即用）

### 📊 完整的指标支持
- 分类：Accuracy, Precision, Recall, F1
- 分割：mIoU, Pixel Accuracy
- 检测：COCO-style mAP@0.5, mAP@0.5:0.95

### 🎯 内置任务支持
- ✅ 图像分类
- ✅ 目标检测（COCO/VOC 格式）
- ✅ 语义分割（Cityscapes/ADE20K）

---

### 🏗️ 架构概览（v0.3.0）

```
dvisionix/
├── registry.py          # 全局注册表（MODELS / TASKS / LOSSES / METRICS / DATASETS / ...）
├── config/               # Config（YAML 继承 / CLI 覆盖 / schema 校验）
├── models/
│   ├── base.py          # BaseModel（TASK_TYPES 校验 / init_weights / from_config）
│   ├── layers/          # 自定义层 + timm 层封装（ConvNormAct / SE / MLP / DropPath）
│   ├── backbones/       # TimmBackbone / TimmClassifier / SequentialBackbone
│   ├── necks/           # FPN / PANet
│   ├── heads/           # 分类（Cls/ArcFace/MultiLabel）· 分割（Seg/FCN/DeepLabV3/UNet）· 检测（Det/FCOS/RetinaNet）
│   ├── detectors/       # SingleStageDetector + FCOS（anchor-free）+ RetinaNet（anchor-based）
│   ├── classifiers/     # LinearClassifier（backbone + 分类头组合）
│   ├── segmenters/      # SegmentationModel（backbone + 分割头组合）
│   ├── toy/             # 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）
│   └── losses/          # BaseLoss + LossComposer + 检测 assigner/损失（即插即用）
├── training/
│   ├── trainer.py       # 统一 Trainer（Task 驱动 / DDP / AMP / 梯度累积 / resume / work_dir）
│   ├── tasks/           # BaseTask + 分类/检测/分割任务（optimizer/loss/metrics 全配置化）—— 主扩展点
│   ├── callbacks/       # CallbackList + ProgressBar/ModelCheckpoint/EarlyStopping —— 主扩展点
│   ├── optim/           # OPTIMIZERS / SCHEDULERS 注册表 + build_optimizer / build_scheduler
│   ├── workdir.py       # 工作目录隔离 + resume 三态 + config dump
│   └── builder.py       # build_callbacks / build_trainer
├── data/
│   ├── sample.py         # Sample 协议 + ImageMode / ImageInfo / NormalizationSpec
│   ├── base.py          # BaseDataset（Sample 驱动 + mask 路径加载）
│   ├── collate.py       # detection_collate / segmentation_collate
│   ├── presets.py       # 主流公开数据集工具箱（CIFAR/ImageNet/COCO/VOC/Cityscapes/ADE20K/ImageFolder）
│   ├── datasets/
│   │   └── custom.py        # CustomDataset（自定义模板范例）
│   └── transforms/
│       ├── base.py        # BaseTransform / TransformPipeline
│       ├── image.py       # 原子：ImageResize / Flip / Crop / ColorJitter / Normalize / ToTensor
│       ├── geometric.py   # 几何同步：BoxSyncResize / BoxSyncRandomHorizontalFlip / BoxSyncRandomCrop / BoxSyncPad
│       ├── labels.py      # LabelToTensor / BoxesToTensor / MaskToTensor
│       ├── third_party.py # AlbumentationsWrapper（适配 albumentations / kornia / torchvision）
│       └── builder.py     # build_transform / build_pipeline
├── metrics/
│   ├── classification.py  # Accuracy / Precision / Recall / F1
│   ├── segmentation.py    # mIoU / Pixel Accuracy
│   ├── detection.py       # COCO mAP（内置实现 + torchmetrics 可选）
│   ├── collection.py      # MetricCollection 组合容器（原子指标自由组合）
│   └── presets.py         # get_preset_metrics + 预设组合类
└── utils/
    └── logging/           # 日志/可视化（console + file + JSONL + TensorBoard）
```

- **配置驱动**：所有入口统一通过 `tools/train.py --config` 或 `Config` 编程 API。
- **注册表**：`build_model` / `build_task` / `build_loss` / `build_metric` / `build_dataset` 从全局注册表按名称查找。
- **归一化唯一权威**：transforms 内的 `Normalize` 类掌控归一化，`BaseDataset` 自动感知避免二次归一化。
- **Loss 在模型层**：dvisionix/models/losses（BaseLoss 继承 + LossComposer 自由组合，接入任意 Task）。
- **日志/可视化**：统一在 utils/logging（TrainingLogger），支持 console / 文件 / JSONL / TensorBoard。
- **工作目录与续训**：默认 ~/dvisionix_runs/&lt;exp&gt;/&lt;ts&gt;（代码库外），--resume auto 自动续训；支持 DDP 多卡。

---

## 🚀 快速开始


### 安装

`ash
# 克隆项目
git clone https://github.com/LAZYzhaoyang/DVisionix.git
cd DVisionix

# 创建 conda 环境
conda create -n dvisionix python=3.10 -y
conda activate dvisionix

# 安装 PyTorch（根据你的 CUDA 版本）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
pip install opencv-python numpy pyyaml tensorboard matplotlib
`

### 3 分钟上手：Config 驱动训练

```bash
# 使用合成数据快速验证分类 pipeline
python tools/train.py --config configs/classification/demo_synthetic.yaml

# 覆盖单个参数
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --cfg-options training.num_epochs=5 training.optimizer.lr=0.01

# 自动续训最近一次任务（work_dir 内自动查找 last.pt）
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --resume auto
```

**编程方式：**

```python
from dvisionix.config import Config
from dvisionix.models import build_model
from dvisionix.training import Trainer, build_task

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())
task = build_task(cfg.task.to_dict() if "task" in cfg else {"type": "ClassificationTask", "num_classes": cfg.model.num_classes})
trainer = Trainer(task=task, train_loader=train_loader, val_loader=val_loader,
                  max_epochs=cfg.training.num_epochs, amp=True, seed=42)
trainer.fit(model)
```


---

## 📖 文档索引

| 文档 | 说明 |
|------|------|
| [快速开始](docs/quick_start.md) | 安装、Config 驱动训练、编程式训练 |
| [配置系统](docs/config_system.md) | Config 加载 / 继承 / CLI 覆盖 / schema 校验 |
| [数据模块](docs/data.md) | Sample 协议、BaseDataset、transforms、数据集工具箱 |
| [骨干网络（timm）](docs/backbones.md) | TimmBackbone / TimmClassifier |
| [自定义 Layer 与 Model](docs/custom_models.md) | layers、注册与配置驱动组装 |
| [指标（Metrics）](docs/metrics.md) | 原子指标、MetricCollection、预设组合 |
| [日志系统](docs/logging.md) | TrainingLogger / JSONL / TensorBoard |
| [模型导出（ONNX）](docs/model_export.md) | ONNXExporter 导出与精度验证 |
| [目标检测](docs/detection.md) | 网格检测器、collate、mAP 评估 |
| [语义分割](docs/segmentation.md) | 分割数据格式与端到端训练 |

---

## 🏗️ 架构设计

### 核心组件

`
┌─────────────────────────────────────────────────────────────┐
│                    Trainer (通用训练引擎)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  Epoch Loop  │  │  Callback  │  │  Metrics  │ │
│  │  Batch Loop  │  │  System   │  │  Collector │ │
│  └───────────┘  └───────────┘  └───────────┘ │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   Task (任务逻辑)                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ training_step  │  │ validation_step  │  │ configure │ │
│  │   (forward + loss) │  │  (eval only)    │  │  optimizer  │ │
│  └───────────────┘  └───────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
`

### 模块依赖

`
data (BaseDataset, build_dataset, Transforms)
    ↓
models (BaseModel, CNN, Segmentation, Detection, Losses)
    ↓
training (Trainer, Tasks, Callbacks)
    ↓
metrics (原子指标 + MetricCollection + 预设组合)
`

---

## 🎯 自定义任务示例

DVisionix 最强大的功能是支持完全自定义的任务逻辑。只要实现三个方法，就可以用通用 Trainer 训练：

`python
from dvisionix.training import BaseTask, Trainer

class MyAIModelTask(BaseTask):
    """自定义 AI 任务"""
    
    def __init__(self, my_param=0.5):
        super().__init__()
        self.my_param = my_param
        self.loss_fn = YourCustomLoss()
    
    def training_step(self, model, batch, device):
        """自定义单步训练逻辑"""
        # 获取数据
        x = batch["your_data"].to(device)
        target = batch["your_target"].to(device)
        
        # 前向传播
        output = model(x)
        
        # 计算损失（你想怎么算就怎么算
        loss = self.loss_fn(output, target)
        
        # 你可以返回任意指标，都会被自动记录
        return {
            "loss": loss,
            "your_metric1": metric1,
            "your_metric2": metric2,
        }
    
    def validation_step(self, model, batch, device):
        """自定义单步验证逻辑"""
        x = batch["your_data"].to(device)
        target = batch["your_target"].to(device)
        
        with torch.no_grad():
            output = model(x)
            loss = self.loss_fn(output, target)
        
        return {"loss": loss}
    
    def configure_optimizers(self, model):
        """自定义优化器配置"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        
        # 支持多种返回格式
        return optimizer  # 或 (optimizer, scheduler)
`

然后直接用 Trainer 训练：

`python
trainer = Trainer(
    task=MyAIModelTask(my_param=0.7),
    train_loader=your_train_loader,
    val_loader=your_val_loader,
    max_epochs=100,
)

trainer.fit(your_model)
`

---

## 📁 项目结构

```
DVisionix/
├── dvisionix/                  # 核心库代码
│   ├── registry.py             # 全局注册表 + build_from_cfg
│   ├── config/                 # Config（YAML 继承 / CLI 覆盖）
│   ├── data/                   # BaseDataset / Sample 契约 / transforms / presets / collate
│   ├── models/                 # BaseModel / backbones / necks / heads / detectors
│   │   └── losses/             # Loss 组件（BaseLoss 继承 + LossComposer 组合）
│   ├── training/               # Trainer / tasks / callbacks / optim / workdir
│   ├── metrics/                # 原子指标 + MetricCollection 组合
│   ├── utils/logging/          # 日志/可视化（console + file + JSONL + TensorBoard）
│   └── export/                 # ONNX 导出
├── tools/train.py              # 配置驱动训练入口
├── configs/                    # 任务示例配置
├── tests/                      # pytest 测试
└── demos/                      # 演示脚本
```

---

## 🧪 测试

`ash
# 运行所有测试
pytest tests/ -v

# 只运行训练模块测试
pytest tests/test_training/ -v

# 只运行数据模块测试
pytest tests/test_data/ -v
`

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 📌 引用

如果你觉得这个项目对你有帮助，欢迎给个 Star ⭐

---

## 📞 问题反馈

如有问题或建议，请在 GitHub 提交 Issue。

---

## 📌 版本演进

### 0.2.0（组件化重构）
- 引入全局注册表与配置驱动入口 `tools/train.py`；模型组件化（backbone/neck/head）；
  `BaseDataset` 归一化交给 transforms（唯一权威）。

### 0.3.0（训练子系统重构）
破坏性变更：
- **Loss 迁移到模型层**：`dvisionix.models.losses`（`BaseLoss` 继承 + `LossComposer` 组合），删除 `dvisionix.training.losses`。
- **Task 全面配置化**：optimizer / scheduler / loss / metrics 均由配置驱动；
  `TensorBoardLogger`、`LearningRateScheduler` 回调与 `utils.visualization.Visualizer` 已移除，日志统一走 `utils.logging.TrainingLogger`。
- **训练能力**：验证指标（acc / mAP / mIoU）进入 epoch 日志；DDP 多卡；work_dir 隔离（默认 `~/dvisionix_runs/<exp>/<ts>`，代码库外）；`--resume auto` 自动续训。

### 0.4.0（模型模块丰富）
- 删除 GeneralizedModel（三任务万能模型，契约弱、检测半成品），替换为具体模型 + 共享脚手架；
- 检测：`FCOSDetector`（anchor-free）与 `RetinaNetDetector`（anchor-based）并存，
  assigner 即插即用（FCOS / MaxIoU / ATSS），decode 进 postprocess；
- 分割：`SegmentationModel` + `DeepLabV3Head`(ASPP) / `UNetDecoder` / `FCNHead`；
- 分类：`LinearClassifier` + `ArcFaceHead` / `MultiLabelHead`（+ BCE 多标签损失）；
- 颈部：新增 `PANet`（FPN + 自底向上路径）；
- 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）迁入 `models.toy`；
- 配置系统支持 `_delete_: true` 替换语义（解决继承时类型专属参数残留）。
- `training/` 重组为 `tasks/` `callbacks/` `optim/` 子包（顶层 API 不变）；
- Config 新增 schema 校验（未知键告警 / 类型校验），CLI 支持 list/dict；
- `ONNXExporter` 支持多输入 / 多输出 / dict 输出 / `backend=trace|dynamo` / 归一化元数据。