import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseHead(nn.Module, ABC):
    """模型头部基类（抽象基类）

    要求子类必须实现：
    - forward: 前向传播方法
    - get_output_sizes: 获取输出尺寸(用于任务头适配)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播方法

        Args:
            features: 骨干网络输出的特征图 (B, C, H, W) 或多尺度特征列表

        Returns:
            任务输出(如分类logits、检测框、分割掩码等)
        """
        pass

    @abstractmethod
    # 任务头基类，要求子类实现forward和get_output_sizes@abstractmethod
    def feature_size(self) -> List[Tuple[int, int, int]]:
        """获取输出尺寸列表（用于后处理适配）

        Returns:
            输出尺寸列表，每个元素为 (C, H, W) 元组
        """