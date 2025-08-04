import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseBackbone(nn.Module, ABC):
    # 骨干网络基类，要求子类实现forward和get_feature_sizes
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播方法

        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            输出张量/特征图
        """
        pass

    @abstractmethod
    def feature_size(self) -> List[Tuple[int, int, int]]:
        """获取输出特征尺寸列表

        Returns:
            特征尺寸列表，每个元素为 (C, H, W) 元组
            单特征返回 [ (C, H, W) ]，多特征返回 [ (C1,H1,W1), (C2,H2,W2), ... ]
        """