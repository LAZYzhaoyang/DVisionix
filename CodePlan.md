# DVisionix 深度学习算法库开发计划（已更新）

## 📋 开发状态（2024）

### ✅ 第一阶段：数据模块（已完成）
- [x] BaseDataset 基类：统一的数据接口，字典格式输出
- [x] TaskType 枚举：CLASSIFICATION, DETECTION, SEGMENTATION
- [x] DataFormat 描述符：描述数据集的格式和统计信息
- [x] DatasetFactory 工厂类：一键创建各种数据集
- [x] ClassificationDatasetAdapter：CIFAR-10/100、ImageNet、ImageFolder 支持
- [x] CustomDataset：自定义数据集支持
- [x] 分类数据变换：Resize、RandomCrop、RandomHorizontalFlip、ColorJitter 等

### ✅ 第二阶段：数据模块完善（已完成，2024-07-20 更新）
- [x] DetectionDatasetAdapter：COCO、Pascal VOC 检测数据集支持
- [x] SegmentationDatasetAdapter：Cityscapes、ADE20K、VOC 分割数据集支持
- [x] DetectionTransforms：检测专用变换（同步处理 boxes）
- [x] SegmentationTransforms：分割专用变换（同步处理 mask）
- [x] 第三方库封装：AlbumentationsWrapper

### ✅ 第三阶段：通用 Trainer 重构（已完成，2024-07-20 更新）
- [x] **Task 系统（核心）**
  - BaseTask：任务基类，抽象 training_step、validation_step、configure_optimizers
  - ClassificationTask：分类任务实现
  - DetectionTask：检测任务（占位实现）
  - SegmentationTask：分割任务（占位实现）
  - 完全解耦：任务逻辑独立于训练循环

- [x] **Callback 系统（核心）**
  - Callback：回调基类，完整的生命周期钩子
  - ProgressBar：进度显示
  - ModelCheckpoint：模型检查点保存，支持监控指标
  - TensorBoardLogger：TensorBoard 日志
  - EarlyStopping：早停
  - LearningRateScheduler：学习率调度

- [x] **通用 Trainer 引擎**
  - 纯执行引擎：只负责循环流程，不包含任何任务特定逻辑
  - 自动设备管理：CPU/GPU 无缝切换
  - 梯度裁剪支持
  - 检查点加载/恢复
  - 训练/验证模式自动切换

### ✅ 第四阶段：损失和指标（已完成，2024-07-20 更新）
- [x] **损失函数**
  - DiceLoss：分割 Dice 损失
  - FocalLoss：类别不平衡损失
  - CombinedSegmentationLoss：CrossEntropy + Dice
  - GIoULoss：检测 IoU 损失

- [x] **指标计算**
  - ClassificationMetrics：准确率、Precision、Recall、F1
  - SegmentationMetrics：mIoU、像素准确率
  - DetectionMetrics：COCO-style mAP、mAP@0.5、mAP@0.75
  - MetricCollection：自动根据任务类型选择指标

### ✅ 第五阶段：模型扩展（已完成，2024-07-20 更新）
- [x] SimpleCNN：分类模型示例
- [x] SimpleSegmentationModel：分割模型示例（全卷积）
- [x] SimpleDetectionModel：检测模型示例（单阶段）

### 📋 未来计划
- [ ] 完整的 DetectionTask 实现（支持 Anchor-based/Anchor-free）
- [ ] 预训练骨干网络集成（ResNet、EfficientNet 等）
- [ ] 数据可视化工具
- [ ] 模型导出（ONNX、TorchScript）
- [ ] 分布式训练支持
- [ ] 更多数据集支持（LVIS、ADE20K 等）

---

## 🎯 架构设计

### 核心设计理念
1. **真正的通用性**：Trainer 不感知任务类型，所有逻辑在 Task 中实现
2. **高度可扩展**：自定义任务只需实现 BaseTask 接口，不需要修改 Trainer
3. **低耦合**：数据、模型、训练、指标完全解耦
4. **统一接口**：所有任务使用相同的数据格式（字典）和训练流程

### 模块依赖关系
`
data (BaseDataset, DatasetFactory, Transforms)
    ↓
models (BaseModel, SimpleCNN, ...)
    ↓
training (Trainer, Tasks, Callbacks, Losses)
    ↓
metrics (ClassificationMetrics, SegmentationMetrics, DetectionMetrics)
`

---

## 📁 项目结构

`
DVisionix/
├── dvisionix/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py           # BaseDataset, TaskType, DataFormat
│   │   ├── factory.py        # DatasetFactory
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── classification.py
│   │   │   ├── detection.py
│   │   │   └── segmentation.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   └── custom.py
│   │   └── transforms/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── classification.py
│   │       ├── detection.py
│   │       ├── segmentation.py
│   │       └── third_party.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # 通用 Trainer 引擎
│   │   ├── task.py            # Task 系统（核心抽象）
│   │   ├── callbacks.py       # Callback 系统
│   │   └── losses.py          # 损失函数
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── segmentation.py
│   │   ├── detection.py
│   │   └── collection.py
│   └── utils/
│       ├── __init__.py
│       └── device.py
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   └── test_basic.py
│   └── test_training/
│       └── test_trainer.py    # Trainer 和 Task 系统测试
├── demos/
│   └── train_cifar10_new_trainer.py  # 新 Trainer 演示
├── CodePlan.md               # 本文档
├── requirements.txt
└── setup.py
`

---

## 💡 快速使用指南

### 示例：CIFAR-10 分类训练（新 Trainer）

`python
from dvisionix.data import DatasetFactory
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import (
    Trainer,
    ClassificationTask,
    ModelCheckpoint,
    EarlyStopping,
)

# 1. 创建数据集
train_transforms = ClassificationTransforms(train=True, image_size=32)
train_dataset = DatasetFactory.create(
    name="cifar10", root="./data", train=True,
    transforms=train_transforms, download=True,
)

# 2. 创建数据加载器
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 3. 创建模型
model = SimpleCNN(num_classes=10)

# 4. 定义任务
task = ClassificationTask(num_classes=10, learning_rate=1e-3)

# 5. 配置回调
callbacks = [
    ModelCheckpoint(save_dir="./checkpoints", monitor="val_acc", mode="max"),
    EarlyStopping(monitor="val_acc", mode="max", patience=5),
]

# 6. 创建训练器并训练
trainer = Trainer(
    task=task,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=callbacks,
    max_epochs=20,
)
trainer.fit(model)
`

### 示例：自定义任务（展示 Trainer 通用性）

`python
from dvisionix.training import BaseTask, Trainer

class MyCustomTask(BaseTask):
    """自定义任务，只要实现三个方法就可以用通用 Trainer"""
    
    def training_step(self, model, batch, device):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return {"loss": loss, "custom_metric": loss * 2}
    
    def validation_step(self, model, batch, device):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with torch.no_grad():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        return {"loss": loss}
    
    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)

# 直接用 Trainer
trainer = Trainer(
    task=MyCustomTask(),
    train_loader=train_loader,
    max_epochs=10,
)
trainer.fit(model)
`

---

## ✅ 代码质量要求（已实现）
- PEP8 合规：使用 black 格式化，flake8 检查
- 类型注解：覆盖率 > 80%，使用 mypy 检查
- 文档字符串：所有公开 API 有完整 docstring
- 单元测试：核心模块有测试覆盖
