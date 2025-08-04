import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

class BaseLayer(nn.Module, ABC):
    # 网络层基类，要求子类实现forward
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """前向传播方法

        Args:
            x: 输入张量
            *args: 其他位置参数
            **kwargs: 其他关键字参数

        Returns:
            输出张量
        """
