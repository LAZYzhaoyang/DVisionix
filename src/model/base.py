"""
统一的基础模型类
提供统一的特征尺寸计算、信息获取和兼容性检查功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union


class BaseModule(nn.Module, ABC):
    """
    统一的基础模块类
    所有模型组件的基类，提供统一的尺寸管理和信息获取功能
    """
    
    def __init__(self, input_size: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None):
        """
        初始化基础模块
        
        Args:
            input_size: 输入特征尺寸，可以是单个(C, H, W)元组或多个输入的列表
        """
        super().__init__()
        
        # 输入尺寸信息
        self._input_size = input_size
        self._output_size = None
        
        # 如果提供了输入尺寸，自动计算输出尺寸
        if input_size is not None:
            self._calculate_feature_sizes()
    
    def _calculate_feature_sizes(self):
        """
        基于前向推理计算特征尺寸
        使用随机张量进行前向传播，记录输出尺寸
        """
        try:
            # 创建批量数据（批次大小=1）
            if isinstance(self._input_size, tuple):
                # 单个输入：添加批次维度
                batch_input = torch.randn(1, *self._input_size)
                output = self.forward(batch_input)
                
                # 处理不同类型的输出
                if isinstance(output, torch.Tensor):
                    self._output_size = tuple(output.shape[1:])  # 去除批次维度
                elif isinstance(output, (list, tuple)):
                    self._output_size = [tuple(tensor.shape[1:]) for tensor in output]
                else:
                    raise ValueError(f"不支持的输出类型: {type(output)}")
                    
            elif isinstance(self._input_size, list):
                # 多个输入：为每个输入添加批次维度
                batch_inputs = [torch.randn(1, *size) for size in self._input_size]
                output = self.forward(*batch_inputs)
                
                # 处理不同类型的输出
                if isinstance(output, torch.Tensor):
                    self._output_size = tuple(output.shape[1:])  # 去除批次维度
                elif isinstance(output, (list, tuple)):
                    self._output_size = [tuple(tensor.shape[1:]) for tensor in output]
                else:
                    raise ValueError(f"不支持的输出类型: {type(output)}")
                    
        except Exception as e:
            raise ValueError(f"无法计算特征尺寸，输入尺寸可能不兼容: {e}")
    
    @abstractmethod
    def forward(self, *inputs):
        """前向传播方法，必须由子类实现"""
        pass
    
    def set_input_size(self, input_size: Union[Tuple[int, int, int], List[Tuple[int, int, int]]]):
        """
        设置输入尺寸并重新计算输出尺寸
        
        Args:
            input_size: 新的输入尺寸
        """
        self._input_size = input_size
        if input_size is not None:
            self._calculate_feature_sizes()
    
    def get_input_size(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        """获取输入尺寸"""
        return self._input_size
    
    def get_output_size(self) -> Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
        """获取输出尺寸"""
        return self._output_size
    
    def info(self) -> str:
        """
        获取模块的详细信息
        
        Returns:
            包含模块信息的字符串
        """
        info_lines = [
            f"模块类型: {self.__class__.__name__}",
            f"输入尺寸: {self._input_size}",
            f"输出尺寸: {self._output_size}",
            f"参数数量: {sum(p.numel() for p in self.parameters())}",
        ]
        
        # 添加特定模块的额外信息
        if hasattr(self, 'num_layers'):
            info_lines.append(f"层数: {self.num_layers}")
        if hasattr(self, 'in_channels'):
            info_lines.append(f"输入通道数: {self.in_channels}")
        if hasattr(self, 'out_channels'):
            info_lines.append(f"输出通道数: {self.out_channels}")
            
        return "\n".join(info_lines)
    
    def check_compatibility(self, prev_module: Optional['BaseModule'] = None) -> bool:
        """
        检查与前一模块的兼容性
        
        Args:
            prev_module: 前一模块实例
            
        Returns:
            是否兼容
        """
        if prev_module is None:
            return True
            
        prev_output = prev_module.get_output_size()
        current_input = self.get_input_size()
        
        # 如果任一模块没有尺寸信息，无法检查兼容性
        if prev_output is None or current_input is None:
            return True
            
        # 检查单个输入的情况
        if isinstance(current_input, tuple) and isinstance(prev_output, tuple):
            return current_input == prev_output
        
        # 检查多个输入的情况
        elif isinstance(current_input, list) and isinstance(prev_output, list):
            return len(current_input) == len(prev_output) and all(
                curr_in == prev_out for curr_in, prev_out in zip(current_input, prev_output)
            )
        
        # 检查混合情况
        elif isinstance(current_input, list) and isinstance(prev_output, tuple):
            return len(current_input) == 1 and current_input[0] == prev_output
        elif isinstance(current_input, tuple) and isinstance(prev_output, list):
            return len(prev_output) == 1 and current_input == prev_output[0]
        
        return False
    
    def __repr__(self):
        """模块的字符串表示"""
        return f"{self.__class__.__name__}(input_size={self._input_size}, output_size={self._output_size})"


# 为不同类型的模块提供特定接口（保持向后兼容）
class BaseLayer(BaseModule):
    """基础层类"""
    pass


class BaseBackbone(BaseModule):
    """基础骨干网络类"""
    pass


class BaseNeck(BaseModule):
    """基础颈部网络类"""
    pass


class BaseHead(BaseModule):
    """基础头部网络类"""
    pass