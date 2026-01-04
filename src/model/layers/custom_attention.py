"""
自定义窗口注意力层
基于新的统一基类实现
"""

import torch
import torch.nn as nn
from ..base import BaseLayer


class CustomWindowAttention(BaseLayer):
    """
    自定义窗口注意力层
    实现基于窗口的注意力机制
    """
    
    def __init__(self, in_channels: int, window_size: int = 7, num_heads: int = 8, 
                 input_size: tuple = None):
        """
        初始化窗口注意力层
        
        Args:
            in_channels: 输入通道数
            window_size: 窗口大小
            num_heads: 注意力头数
            input_size: 输入特征尺寸 (C, H, W)
        """
        super().__init__(input_size)
        
        self.in_channels = in_channels
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 注意力机制的核心组件
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        
        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 初始化相对位置编码索引
        self._init_relative_position_index()
    
    def _init_relative_position_index(self):
        """初始化相对位置编码索引"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            输出张量，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 确保输入通道数匹配
        if C != self.in_channels:
            raise ValueError(f"输入通道数不匹配: 期望 {self.in_channels}, 实际 {C}")
        
        # 将特征图分割为窗口
        x = x.view(B, C, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        
        # 计算QKV
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, 
                                  self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, 
               self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 应用softmax
        attn = attn.softmax(dim=-1)
        
        # 注意力加权
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        
        # 投影
        x = self.proj(x)
        
        # 恢复原始形状
        x = x.view(-1, self.window_size, self.window_size, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, H // self.window_size, W // self.window_size, 
                   C, self.window_size, self.window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        
        return x