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
- DatasetFactory 一键创建标准数据集

### 📊 完整的指标支持
- 分类：Accuracy, Precision, Recall, F1
- 分割：mIoU, Pixel Accuracy
- 检测：COCO-style mAP@0.5, mAP@0.5:0.95

### 🎯 内置任务支持
- ✅ 图像分类
- ✅ 目标检测（COCO/VOC 格式）
- ✅ 语义分割（Cityscapes/ADE20K）

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

### 3 分钟上手：CIFAR-10 分类

`python
import torch
from torch.utils.data import DataLoader

# 1. 导入 DVisionix
from dvisionix.data import DatasetFactory
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import (
    Trainer,
    ClassificationTask,
    ModelCheckpoint,
    EarlyStopping,
)

# 2. 创建数据集
train_transforms = ClassificationTransforms(train=True, image_size=32)
val_transforms = ClassificationTransforms(train=False, image_size=32)

train_dataset = DatasetFactory.create(
    name="cifar10",
    root="./data",
    train=True,
    transforms=train_transforms,
    download=True,
)

val_dataset = DatasetFactory.create(
    name="cifar10",
    root="./data",
    train=False,
    transforms=val_transforms,
    download=True,
)

# 3. 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

# 4. 创建模型和任务
model = SimpleCNN(num_classes=10)

task = ClassificationTask(
    num_classes=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# 5. 配置回调
callbacks = [
    ModelCheckpoint(
        save_dir="./checkpoints/cifar10",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
    ),
    EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
        restore_best_weights=True,
    ),
]

# 6. 创建训练器并开始训练
trainer = Trainer(
    task=task,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=callbacks,
    device="auto",
    max_epochs=20,
)

trainer.fit(model)
`

---

## 📖 文档索引

| 文档 | 说明 |
|------|------|
| [快速入门](docs/quick_start.md) | 安装和基础使用教程 |
| [数据模块指南](docs/data_module.md) | 数据集、适配器、变换完整用法 |
| [训练系统指南](docs/training_module.md) | Trainer、Task、Callback 系统详解 |
| [自定义任务教程](docs/custom_task.md) | 如何实现自己的训练任务 |
| [指标计算](docs/metrics.md) | 各任务指标计算详解 |
| [API 参考](docs/api_reference.md) | 完整的 API 文档 |

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
data (BaseDataset, DatasetFactory, Transforms)
    ↓
models (BaseModel, CNN, Segmentation, Detection)
    ↓
training (Trainer, Tasks, Callbacks, Losses)
    ↓
metrics (ClassificationMetrics, SegmentationMetrics, DetectionMetrics)
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

`
DVisionix/
├── dvisionix/                  # 核心库代码
│   ├── __init__.py
│   ├── data/                     # 数据模块
│   │   ├── base.py              # BaseDataset, TaskType, DataFormat
│   │   ├── factory.py           # DatasetFactory
│   │   ├── adapters/           # 标准数据集适配器
│   │   │   ├── classification.py
│   │   │   ├── detection.py
│   │   │   └── segmentation.py
│   │   ├── datasets/            # 自定义数据集
│   │   └── transforms/          # 数据变换
│   ├── models/                   # 模型模块
│   │   └── base.py
│   ├── training/                 # 训练模块（核心架构）
│   │   ├── trainer.py           # 通用 Trainer
│   │   ├── task.py              # Task 系统
│   │   ├── callbacks.py         # Callback 系统
│   │   └── losses.py           # 损失函数
│   ├── metrics/                  # 指标模块
│   │   ├── classification.py
│   │   ├── segmentation.py
│   │   └── detection.py
│   └── utils/                     # 工具函数
├── docs/                       # 文档
├── tests/                      # 测试
├── demos/                      # 演示脚本
├── CodePlan.md                  # 开发计划
└── requirements.txt             # 依赖清单
`

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
