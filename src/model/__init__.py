"""
DVisionix 模型模块
提供统一的模型组件和尺寸管理功能
"""

from .base import BaseModule, BaseLayer, BaseBackbone, BaseNeck, BaseHead

# 导入骨干网络
from .backbone.custom_backbone import CustomBackbone

# 导入层模块
from .layers.custom_attention import CustomWindowAttention

# 导入工具模块
from .utils.size_calculator import (
    calculate_conv_output_size,
    calculate_pool_output_size,
    calculate_upsample_output_size,
    calculate_linear_output_size,
    calculate_sequence_output_size,
    check_size_compatibility,
    print_size_info
)

__all__ = [
    # 基础类
    'BaseModule', 'BaseLayer', 'BaseBackbone', 'BaseNeck', 'BaseHead',
    
    # 骨干网络
    'CustomBackbone',
    
    # 层模块
    'CustomWindowAttention',
    
    # 工具函数
    'calculate_conv_output_size',
    'calculate_pool_output_size', 
    'calculate_upsample_output_size',
    'calculate_linear_output_size',
    'calculate_sequence_output_size',
    'check_size_compatibility',
    'print_size_info'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "DVisionix Team"
__description__ = "统一的深度学习模型组件库，支持自动尺寸管理和兼容性检查"