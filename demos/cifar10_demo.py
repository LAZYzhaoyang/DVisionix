# D:\ZhaoyangProject\DVisionix\demos\cifar10_enhanced_demo.py

"""
DVisionix CIFAR-10 增强版 Demo

包含 TensorBoard 可视化和多指标评估。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("DVisionix CIFAR-10 增强版 Demo (TensorBoard + 多指标)")
print("=" * 60)

from dvisionix.data import DatasetFactory, ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import Trainer
from dvisionix.utils import Visualizer
from dvisionix.metrics import (
    MetricCollection,
    Accuracy,
    TopKAccuracy,
    Precision,
    Recall,
    F1Score,
)

# 配置
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "auto"
NUM_CLASSES = 10

# 创建数据集和模型
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

model = SimpleCNN(num_classes=NUM_CLASSES)

# 创建指标集合
train_metrics = MetricCollection([
    Accuracy(),
    TopKAccuracy(k=5),
])

val_metrics = MetricCollection([
    Accuracy(),
    TopKAccuracy(k=5),
    Precision(num_classes=NUM_CLASSES),
    Recall(num_classes=NUM_CLASSES),
    F1Score(num_classes=NUM_CLASSES),
])

# 创建可视化器
visualizer = Visualizer(
    log_dir="./logs",
    experiment_name="cifar10_demo",
    comment="v1",
)

# 记录超参数
visualizer.log_hparams(
    hparam_dict={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "num_classes": NUM_CLASSES,
    },
    metric_dict={"hparam/dummy": 0.0},
)

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

# 手动训练循环（演示可视化和指标）
import torch
from dvisionix.utils import get_device, move_to_device

device = get_device(DEVICE)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

global_step = 0

for epoch in range(NUM_EPOCHS):
    # 训练
    model.train()
    train_metrics.reset()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = move_to_device(batch, device)
        images, labels = batch["image"], batch["label"]
        
        # 前向
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新指标
        train_metrics.update(outputs, labels)
        
        # 记录到 TensorBoard
        visualizer.log_scalar("train/batch_loss", loss.item(), global_step)
        global_step += 1
        
        # 记录图像样本（第一个 batch）
        if batch_idx == 0:
            visualizer.log_images("train/samples", images[:8], epoch, normalize=True)
        
        if (batch_idx + 1) % 100 == 0:
            train_result = train_metrics.compute()
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {train_result['accuracy']:.2f}%"
            )
    
    # 训练 epoch 结束
    train_result = train_metrics.compute()
    visualizer.log_scalar("train/epoch_loss", loss.item(), epoch)
    visualizer.log_scalar("train/accuracy", train_result["accuracy"], epoch)
    visualizer.log_scalar("train/top5_accuracy", train_result["top5_accuracy"], epoch)
    
    # 验证
    model.eval()
    val_metrics.reset()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch, device)
            images, labels = batch["image"], batch["label"]
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_metrics.update(outputs, labels)
    
    val_result = val_metrics.compute()
    val_loss /= len(val_loader)
    
    # 记录验证指标
    visualizer.log_scalar("val/loss", val_loss, epoch)
    for name, value in val_result.items():
        visualizer.log_scalar(f"val/{name}", value, epoch)
    
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] 验证结果:")
    for name, value in val_result.items():
        print(f"  {name}: {value:.2f}%")
    print()

# 训练完成
visualizer.close()

print("=" * 60)
print("✅ 训练完成！")
print(f"\nTensorBoard 日志已保存到: ./logs")
print(f"启动 TensorBoard 查看训练曲线:")
print(f"  tensorboard --logdir ./logs")
print("=" * 60)