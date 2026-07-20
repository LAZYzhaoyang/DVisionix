# D:\\ZhaoyangProject\\DVisionix\\dvisionix\\models\\base.py

"""
模型基类和示例模型

提供统一的模型接口和简单的示例模型，用于快速验证数据流程。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class BaseModel(nn.Module):
    """
    所有模型的基类
    
    提供统一的接口规范、输入输出检查、参数统计等功能。
    """
    
    def __init__(self):
        super().__init__()
        self.task_type: Optional[str] = None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Any:
        """前向传播，子类必须实现"""
        raise NotImplementedError
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_device(self) -> torch.device:
        """获取模型所在的设备"""
        return next(self.parameters()).device


class SimpleCNN(BaseModel):
    """
    简单的 CNN 分类模型（用于演示和测试）
    
    适用于 CIFAR-10 等小尺寸图像的分类任务。
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """
        Args:
            num_classes: 分类类别数
            in_channels: 输入图像通道数
        """
        super().__init__()
        self.task_type = "classification"
        self.num_classes = num_classes
        
        # 简单的 CNN 架构
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, height, width)
            
        Returns:
            分类 logits (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.flatten(1)  # 展平
        x = self.classifier(x)
        return x


class SimpleSegmentationModel(BaseModel):
    """
    简单的分割模型（用于演示和测试）
    
    基于全卷积网络，输出与输入相同尺寸的分割图。
    """
    
    def __init__(self, num_classes: int = 21, in_channels: int = 3):
        """
        Args:
            num_classes: 分割类别数
            in_channels: 输入图像通道数
        """
        super().__init__()
        self.task_type = "segmentation"
        self.num_classes = num_classes
        
        # 编码器 (下采样)
        self.encoder1 = self._make_block(in_channels, 64)
        self.encoder2 = self._make_block(64, 128)
        self.encoder3 = self._make_block(128, 256)
        
        # 解码器 (上采样)
        self.decoder3 = self._make_block(256, 128)
        self.decoder2 = self._make_block(128, 64)
        self.decoder1 = self._make_block(64, 32)
        
        # 分类头
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, height, width)
            
        Returns:
            分割 logits (batch_size, num_classes, height, width)
        """
        input_size = x.shape[2:]
        
        # 编码
        x1 = F.max_pool2d(self.encoder1(x), 2)  # 1/2
        x2 = F.max_pool2d(self.encoder2(x1), 2)  # 1/4
        x3 = F.max_pool2d(self.encoder3(x2), 2)  # 1/8
        
        # 解码
        x = F.interpolate(self.decoder3(x3), scale_factor=2, mode="bilinear", align_corners=True)
        x = F.interpolate(self.decoder2(x), scale_factor=2, mode="bilinear", align_corners=True)
        x = F.interpolate(self.decoder1(x), scale_factor=2, mode="bilinear", align_corners=True)
        
        # 确保输出尺寸与输入一致
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        
        x = self.final_conv(x)
        return x


class SimpleDetectionModel(BaseModel):
    """
    简单的检测模型（用于演示和测试）
    
    基于单阶段检测器，输出分类和边界框回归。
    注意：这只是一个简化示例，实际使用建议用 Detectron2 或 MMDetection。
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        in_channels: int = 3,
        num_anchors: int = 9,
    ):
        """
        Args:
            num_classes: 类别数（不含背景）
            in_channels: 输入通道数
            num_anchors: 每个位置的锚框数
        """
        super().__init__()
        self.task_type = "detection"
        self.num_classes = num_classes
        
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 检测头
        # 分类：每个 anchor 预测 num_classes + 1 个类别（包括背景）
        self.cls_head = nn.Conv2d(256, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
        # 回归：每个 anchor 预测 4 个坐标偏移
        self.reg_head = nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, height, width)
            
        Returns:
            包含分类和回归特征的字典
        """
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_reg = self.reg_head(features)
        
        return {
            "cls_logits": cls_logits,
            "bbox_reg": bbox_reg,
        }
