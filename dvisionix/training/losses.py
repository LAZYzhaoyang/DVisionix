# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\training\\losses.py

"""
损失函数模块

提供各种任务的损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_loss_function(task_type: str, **kwargs) -> nn.Module:
    """
    根据任务类型获取默认损失函数
    
    Args:
        task_type: 任务类型 ('classification', 'segmentation', 'detection')
        **kwargs: 其他参数
        
    Returns:
        损失函数实例
    """
    if task_type == "classification":
        return nn.CrossEntropyLoss()
    elif task_type == "segmentation":
        ignore_index = kwargs.get("ignore_index", 255)
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif task_type == "detection":
        # 检测任务通常需要自定义复合损失
        return DetectionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# =============================================================================
# 分割损失函数
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss（适用于分割任务）
    
    基于 Dice 系数的损失，对类别不平衡不敏感。
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 (B, C, H, W)
            targets: 目标标签 (B, H, W)
            
        Returns:
            Dice loss
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # 转换为概率
        probs = F.softmax(logits, dim=1)
        
        # 创建 one-hot 目标
        targets_onehot = torch.zeros_like(probs)
        for c in range(num_classes):
            targets_onehot[:, c] = (targets == c).float()
        
        # 忽略指定类别
        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index).unsqueeze(1)
            targets_onehot = targets_onehot.masked_fill(ignore_mask, 0)
            probs = probs.masked_fill(ignore_mask, 0)
        
        # 计算 Dice 系数
        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets_onehot.view(batch_size, num_classes, -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 取平均
        loss = 1.0 - dice.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    解决类别不平衡问题，降低简单样本的权重，关注困难样本。
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 (B, C, ...) 或 (B, C)
            targets: 目标标签 (B, ...) 或 (B,)
            
        Returns:
            Focal loss
        """
        # 忽略指定类别
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            
            # 应用掩码（只对分割有效，分类需要特殊处理）
            if logits.dim() > 2:
                logits = logits.permute(0, 2, 3, 1).contiguous()
                logits = logits[valid_mask]
                targets = targets[valid_mask]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term
        focal_term = (1.0 - pt).pow(self.gamma)
        
        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term
        
        # 计算最终损失
        loss = focal_term * ce_loss
        
        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    分割任务的组合损失
    
    CrossEntropyLoss + DiceLoss
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice


# =============================================================================
# 检测损失函数
# =============================================================================

class DetectionLoss(nn.Module):
    """
    检测任务的组合损失
    
    分类损失 + 边界框回归损失
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        box_weight: float = 1.0,
        use_smooth_l1: bool = True,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        
        if use_smooth_l1:
            self.box_loss_fn = F.smooth_l1_loss
        else:
            self.box_loss_fn = F.l1_loss
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        matched_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算检测损失
        
        Args:
            pred_logits: 预测分类 logits
            pred_boxes: 预测边界框
            target_labels: 目标标签
            target_boxes: 目标边界框
            matched_mask: 正样本掩码
            
        Returns:
            包含各损失和总损失的字典
        """
        # 分类损失
        if matched_mask is not None:
            cls_loss = F.cross_entropy(pred_logits, target_labels, reduction="mean")
        else:
            cls_loss = F.cross_entropy(pred_logits, target_labels, reduction="mean")
        
        # 边界框损失（只对正样本计算）
        if matched_mask is not None and matched_mask.sum() > 0:
            box_loss = self.box_loss_fn(pred_boxes[matched_mask], target_boxes[matched_mask])
        else:
            box_loss = self.box_loss_fn(pred_boxes, target_boxes)
        
        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss
        
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
        }


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss
    
    比 L1/L2 损失更适合边界框回归，直接优化 IoU。
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 GIoU Loss
        
        Args:
            pred_boxes: 预测框 (N, 4) 格式 [x1, y1, x2, y2]
            target_boxes: 目标框 (N, 4) 格式 [x1, y1, x2, y2]
            
        Returns:
            GIoU loss
        """
        # 计算 IoU
        iou, union = self._iou(pred_boxes, target_boxes)
        
        # 计算最小包围框
        cw = torch.max(pred_boxes[:, 2], target_boxes[:, 2]) - torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        ch = torch.max(pred_boxes[:, 3], target_boxes[:, 3]) - torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        
        c_area = cw * ch + 1e-8  # 避免除以零
        
        # GIoU = IoU - (c_area - union) / c_area
        giou = iou - (c_area - union) / c_area
        
        # Loss = 1 - GIoU
        loss = 1.0 - giou
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 IoU"""
        # 交集
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        intersection = w * h
        
        # 并集
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection + 1e-8
        
        iou = intersection / union
        
        return iou, union
