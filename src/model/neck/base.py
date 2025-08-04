import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseNeck(nn.Module, ABC):
    """模型颈部基类（抽象基类）

    要求子类必须实现：
    - forward: 前向传播方法(处理多尺度特征)
    - get_feature_sizes: 获取颈部输出特征尺寸
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播方法（处理多尺度特征）

        Args:
            features: 骨干网络输出的多尺度特征列表，每个元素为 (B, C, H, W) 张量

        Returns:
            处理后的多尺度特征列表
        """
        pass

    @abstractmethod
    def feature_size(self) -> List[Tuple[int, int, int]]:
        """获取颈部输出特征尺寸列表

        Returns:
            特征尺寸列表，每个元素为 (C, H, W) 元组
        """
        pass

    # 模型颈部基类，要求子类实现forward（处理多尺度特征）和get_feature_sizes
