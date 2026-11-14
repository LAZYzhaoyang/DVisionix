# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 分类组合模型子包（每类一个文件）。
"""分类组合模型子包（每类一个文件）。"""

from .linear import LinearClassifier
from .multi_task import MultiTaskClassifier

__all__ = ["LinearClassifier", "MultiTaskClassifier"]
