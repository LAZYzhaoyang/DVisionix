# D:\\ZhaoyangProject\\DVisionix\\demos\\train_cifar10_new_trainer.py

"""
使用新的通用 Trainer 训练 CIFAR-10 分类任务

演示：
1. 使用 DatasetFactory 创建数据集
2. 使用 ClassificationTask 定义任务逻辑
3. 使用通用 Trainer 进行训练
4. 使用回调系统（检查点、早停）
"""

import sys
sys.path.insert(0, "D:\\ZhaoyangProject\\DVisionix")

import torch
from torch.utils.data import DataLoader

from dvisionix.data import DatasetFactory
from dvisionix.data.transforms import ClassificationTransforms
from dvisionix.models import SimpleCNN
from dvisionix.training import (
    Trainer,
    ClassificationTask,
    ModelCheckpoint,
    EarlyStopping,
)


def main():
    # =====================
    # 配置
    # =====================
    config = {
        "batch_size": 128,
        "num_workers": 0,  # Windows 下建议用 0
        "max_epochs": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "device": "auto",
        "data_root": "./data",
        "save_dir": "./checkpoints/cifar10",
    }
    
    print("=" * 60)
    print("DVisionix: CIFAR-10 分类训练（新 Trainer 演示）")
    print("=" * 60)
    
    # =====================
    # 1. 创建数据
    # =====================
    print("\n[1/4] 创建数据集...")
    
    # 数据变换
    train_transforms = ClassificationTransforms(train=True, image_size=32)
    val_transforms = ClassificationTransforms(train=False, image_size=32)
    
    # 使用工厂创建数据集
    train_dataset = DatasetFactory.create(
        name="cifar10",
        root=config["data_root"],
        train=True,
        transforms=train_transforms,
        download=True,
    )
    
    val_dataset = DatasetFactory.create(
        name="cifar10",
        root=config["data_root"],
        train=False,
        transforms=val_transforms,
        download=True,
    )
    
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["device"] == "auto" or "cuda" in config["device"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    
    # =====================
    # 2. 创建模型
    # =====================
    print("\n[2/4] 创建模型...")
    
    model = SimpleCNN(num_classes=10)
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # =====================
    # 3. 定义任务和回调
    # =====================
    print("\n[3/4] 配置任务和回调...")
    
    # 定义分类任务
    task = ClassificationTask(
        num_classes=10,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # 配置回调
    callbacks = [
        ModelCheckpoint(
            save_dir=config["save_dir"],
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
    
    print("  回调: ModelCheckpoint（监控 val_acc）")
    print("  回调: EarlyStopping（patience=5）")
    
    # =====================
    # 4. 创建训练器并训练
    # =====================
    print("\n[4/4] 开始训练...")
    
    trainer = Trainer(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        device=config["device"],
        max_epochs=config["max_epochs"],
    )
    
    # 开始训练
    trainer.fit(model)
    
    print("\n训练完成！")
    print(f"  最佳验证准确率: {callbacks[0].best_value:.4f}")


def custom_task_example():
    """
    演示如何使用自定义任务
    
    展示 Trainer 的通用性：你可以完全自定义训练逻辑，
    而不需要修改 Trainer 的代码。
    """
    print("\n" + "=" * 60)
    print("自定义任务演示")
    print("=" * 60)
    
    from dvisionix.training import BaseTask
    import torch.nn as nn
    
    class MyCustomTask(BaseTask):
        """我的自定义任务
        
        只要实现三个方法，就可以用通用 Trainer 训练
        """
        
        def __init__(self, lr: float = 1e-3):
            super().__init__()
            self.lr = lr
            self.loss_fn = nn.CrossEntropyLoss()
        
        def training_step(self, model, batch, device):
            """自定义训练步"""
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # 前向传播
            logits = model(x)
            
            # 计算损失
            loss = self.loss_fn(logits, y)
            
            # 计算准确率
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean()
            
            # 返回的字典会被 Trainer 收集用于日志
            return {
                "loss": loss,
                "acc": acc,
                "my_custom_metric": loss * 2,  # 你可以加任意指标
            }
        
        def validation_step(self, model, batch, device):
            """自定义验证步"""
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            with torch.no_grad():
                logits = model(x)
                loss = self.loss_fn(logits, y)
                preds = logits.argmax(dim=1)
                acc = (preds == y).float().mean()
            
            return {
                "loss": loss,
                "acc": acc,
            }
        
        def configure_optimizers(self, model):
            """配置优化器"""
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            # 你可以返回：
            # 1. 单个 optimizer
            # 2. (optimizer, scheduler) 元组
            # 3. 带 monitor 的字典（用于 ReduceLROnPlateau）
            
            return optimizer
    
    print("\n✓ 自定义任务只需要实现三个方法：")
    print("  - training_step: 单步训练逻辑")
    print("  - validation_step: 单步验证逻辑")
    print("  - configure_optimizers: 优化器配置")


if __name__ == "__main__":
    # 运行主训练脚本
    main()
    
    # 演示自定义任务（不实际训练，只展示结构）
    custom_task_example()
