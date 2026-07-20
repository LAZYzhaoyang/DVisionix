# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\training\\task.py

"""
任务接口定义

将任务特定的逻辑（训练步、验证步、优化器配置）完全抽离，
使 Trainer 成为纯执行引擎。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class BaseTask(ABC):
    """
    任务基类
    
    所有训练任务都继承此类，实现任务特定的逻辑。
    
    自定义任务示例：
    >>> class MyCustomTask(BaseTask):
    ...     def training_step(self, model, batch, device):
    ...         x = batch['image'].to(device)
    ...         y = batch['label'].to(device)
    ...         pred = model(x)
    ...         loss = nn.CrossEntropyLoss()(pred, y)
    ...         return {'loss': loss, 'acc': accuracy(pred, y)}
    ...
    ...     def configure_optimizers(self, model):
    ...         return torch.optim.Adam(model.parameters(), lr=1e-3)
    """
    
    @abstractmethod
    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        单步训练逻辑
        
        Args:
            model: 神经网络模型
            batch: 批次数据（字典格式）
            device: 计算设备
            
        Returns:
            包含 'loss' 和其他指标的字典
            - 'loss' 必须存在，用于反向传播
            - 其他键值对会被 Trainer 收集用于日志
        """
        pass
    
    @abstractmethod
    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        单步验证逻辑
        
        Args:
            model: 神经网络模型
            batch: 批次数据（字典格式）
            device: 计算设备
            
        Returns:
            包含验证指标的字典（不需要 loss 梯度）
        """
        pass
    
    @abstractmethod
    def configure_optimizers(self, model: nn.Module) -> Any:
        """
        配置优化器和学习率调度器
        
        Args:
            model: 神经网络模型
            
        Returns:
            支持三种返回格式：
            1. optimizer: 单个优化器
            2. (optimizer, scheduler): 优化器 + 调度器
            3. {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
               （用于 ReduceLROnPlateau）
        """
        pass


# =============================================================================
# 内置标准任务实现
# =============================================================================

class ClassificationTask(BaseTask):
    """
    图像分类任务
    
    输入数据格式要求：
    - batch['image']: 图像张量 (B, C, H, W)
    - batch['label']: 类别标签 (B,)
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_function: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_classes: 类别数量（可选，用于计算准确率）
            learning_rate: 学习率
            weight_decay: 权重衰减
            loss_function: 自定义损失函数，默认使用 CrossEntropyLoss
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function or nn.CrossEntropyLoss()
    
    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # 前向传播
        logits = model(images)
        loss = self.loss_function(logits, labels)
        
        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
        
        return {
            "loss": loss,
            "acc": acc,
        }
    
    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            logits = model(images)
            loss = self.loss_function(logits, labels)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
        
        return {
            "loss": loss,
            "acc": acc,
        }
    
    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # 默认使用 ReduceLROnPlateau 调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class DetectionTask(BaseTask):
    """
    目标检测任务（占位实现，待完善）
    
    输入数据格式要求：
    - batch['image']: 图像张量 (B, C, H, W)
    - batch['boxes']: 边界框列表 [(N1, 4), (N2, 4), ...]
    - batch['labels']: 类别标签列表 [(N1,), (N2,), ...]
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        # TODO: 实现检测训练逻辑
        # 注意：检测模型通常有自己的 loss 计算，输出就是 loss 字典
        images = batch["image"].to(device)
        targets = [
            {
                "boxes": batch["boxes"][i].to(device),
                "labels": batch["labels"][i].to(device),
            }
            for i in range(len(images))
        ]
        
        # 假设模型返回 loss 字典
        loss_dict = model(images, targets)
        
        if isinstance(loss_dict, dict):
            total_loss = sum(loss_dict.values())
            result = {"loss": total_loss}
            result.update({k: v.detach() for k, v in loss_dict.items()})
            return result
        else:
            return {"loss": loss_dict}
    
    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        # TODO: 实现检测验证逻辑（需要计算 mAP）
        images = batch["image"].to(device)
        
        with torch.no_grad():
            # 验证模式下模型通常返回预测框
            outputs = model(images)
        
        # 暂时只返回占位值，需要结合 metrics 模块
        return {"loss": torch.tensor(0.0, device=device)}
    
    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        # 检测任务通常使用 SGD with momentum
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[8, 11],
            gamma=0.1,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class SegmentationTask(BaseTask):
    """
    图像分割任务（占位实现，待完善）
    
    输入数据格式要求：
    - batch['image']: 图像张量 (B, C, H, W)
    - batch['mask']: 分割掩码 (B, H, W)
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
    
    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        logits = model(images)
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(logits, masks)
        
        # 计算像素准确率（忽略 ignore_index）
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            valid_mask = (masks != self.ignore_index)
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == masks[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=device)
        
        return {
            "loss": loss,
            "acc": acc,
        }
    
    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        with torch.no_grad():
            logits = model(images)
            loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)(logits, masks)
            
            preds = logits.argmax(dim=1)
            valid_mask = (masks != self.ignore_index)
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == masks[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=device)
        
        return {
            "loss": loss,
            "acc": acc,
        }
    
    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        # 分割任务通常使用 Adam 或 AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
