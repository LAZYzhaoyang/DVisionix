"""
自定义骨干网络
基于新的统一基类实现
"""

import torch
import torch.nn as nn
from ..base import BaseBackbone


class CustomBackbone(BaseBackbone):
    """
    自定义骨干网络
    实现一个简单的卷积神经网络作为骨干
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_layers: int = 4, input_size: tuple = None):
        """
        初始化自定义骨干网络
        
        Args:
            in_channels: 输入通道数
            base_channels: 基础通道数
            num_layers: 层数
            input_size: 输入特征尺寸 (C, H, W)
        """
        super().__init__(input_size)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建后续卷积层
        self.layers = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(1, num_layers):
            next_channels = current_channels * 2
            layer = nn.Sequential(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, 
                         stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels = next_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            输出张量列表，包含各层的特征图
        """
        # 确保输入通道数匹配
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"输入通道数不匹配: 期望 {self.in_channels}, 实际 {C}")
        
        features = []
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 添加第一个特征图
        
        # 最大池化
        x = self.maxpool(x)
        
        # 后续层
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        return features