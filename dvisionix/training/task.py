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
    目标检测任务（单阶段网格检测器，真正可训练）

    配合 dvisionix.models.GridDetectionModel 使用。模型输出原始张量
    (B, 5 + num_classes, GH, GW)，本任务负责：
    - 中心点目标分配：GT 框中心所在网格单元为正样本；
    - objectness 损失：BCEWithLogits（全网格）；
    - 框回归损失：仅正样本，预测归一化 [cx, cy, w, h] 的 L1；
    - 分类损失：仅正样本，CrossEntropy。

    输入数据格式要求：
    - batch['image']: 图像张量 (B, C, H, W)
    - batch['boxes']: List[Tensor(Ni, 4)]，坐标为像素 [x1, y1, x2, y2]
    - batch['labels']: List[Tensor(Ni,)]，类别索引（0..num_classes-1）
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        obj_weight: float = 1.0,
        box_weight: float = 5.0,
        cls_weight: float = 1.0,
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

    def _compute_losses(self, preds, boxes_list, labels_list, image_hw, device):
        """根据网格预测与 GT 计算损失和统计量。"""
        B, C, GH, GW = preds.shape
        img_h, img_w = image_hw

        obj_logits = preds[:, 0, :, :]                    # (B, GH, GW)
        box_pred = preds[:, 1:5, :, :]                    # (B, 4, GH, GW)
        cls_logits = preds[:, 5:, :, :]                   # (B, num_classes, GH, GW)

        obj_target = torch.zeros((B, GH, GW), device=device)
        box_target = torch.zeros((B, 4, GH, GW), device=device)
        cls_target = torch.full((B, GH, GW), -1, dtype=torch.long, device=device)

        num_pos = 0
        for b in range(B):
            boxes = boxes_list[b].to(device).float()
            labels = labels_list[b].to(device).long()
            for k in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[k]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1).clamp(min=1e-6)
                bh = (y2 - y1).clamp(min=1e-6)
                # 中心所在网格单元
                gx = int((cx / img_w * GW).clamp(0, GW - 1).item())
                gy = int((cy / img_h * GH).clamp(0, GH - 1).item())
                obj_target[b, gy, gx] = 1.0
                # 归一化框目标 [cx, cy, w, h] in [0,1]
                box_target[b, 0, gy, gx] = cx / img_w
                box_target[b, 1, gy, gx] = cy / img_h
                box_target[b, 2, gy, gx] = bw / img_w
                box_target[b, 3, gy, gx] = bh / img_h
                cls_target[b, gy, gx] = labels[k]
                num_pos += 1

        obj_loss = nn.functional.binary_cross_entropy_with_logits(obj_logits, obj_target)

        pos_mask = obj_target > 0.5
        if pos_mask.sum() > 0:
            # 框回归：对正样本单元的 sigmoid(box_pred) 与目标做 L1
            box_pred_s = torch.sigmoid(box_pred)
            pm = pos_mask.unsqueeze(1).expand_as(box_pred_s)
            box_loss = nn.functional.l1_loss(box_pred_s[pm], box_target[pm])

            # 分类：把 (B, C, GH, GW) 变换到 (N_pos, C)
            cls_perm = cls_logits.permute(0, 2, 3, 1)      # (B, GH, GW, C)
            cls_sel = cls_perm[pos_mask]                    # (N_pos, C)
            tgt_sel = cls_target[pos_mask]                  # (N_pos,)
            cls_loss = nn.functional.cross_entropy(cls_sel, tgt_sel)

            with torch.no_grad():
                cls_acc = (cls_sel.argmax(dim=1) == tgt_sel).float().mean()
        else:
            box_loss = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
            cls_acc = torch.tensor(0.0, device=device)

        total = (
            self.obj_weight * obj_loss
            + self.box_weight * box_loss
            + self.cls_weight * cls_loss
        )
        return {
            "loss": total,
            "obj_loss": obj_loss.detach(),
            "box_loss": box_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "cls_acc": cls_acc.detach(),
        }

    def training_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        preds = model(images)
        image_hw = (images.shape[2], images.shape[3])
        return self._compute_losses(preds, batch["boxes"], batch["labels"], image_hw, device)

    def validation_step(self, model: nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        images = batch["image"].to(device)
        with torch.no_grad():
            preds = model(images)
            image_hw = (images.shape[2], images.shape[3])
            result = self._compute_losses(preds, batch["boxes"], batch["labels"], image_hw, device)
        return result

    def configure_optimizers(self, model: nn.Module) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



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
