# -*- coding: utf-8 -*-
"""内置骨干基类：stage 列表 + dry-run 通道推导 + features_only 多尺度输出。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ...registry import BACKBONES
from ..base import BaseModel


@BACKBONES.register()
@BACKBONES.register(name="feature_backbone_base")
class FeatureBackboneBase(BaseModel):
    """内置骨干通用实现：给定 stage 列表，自动推导 out_channels / num_features。

    - features_only=False（默认）: 输出全局池化特征向量 (B, num_features)，适合分类头。
    - features_only=True: 输出多尺度特征图列表，并暴露 out_channels，供 FPN / 检测头 / 分割头使用。
    """

    def __init__(
        self,
        stages: Sequence[nn.Module],
        in_channels: int = 3,
        input_size: int = 32,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("至少需要一个 stage。")
        self.features_only = features_only
        self.stages = nn.ModuleList(stages)
        num_stages = len(self.stages)
        if out_indices is None:
            out_indices = list(range(num_stages))
        self.out_indices = [i % num_stages for i in out_indices]
        stage_channels = self._infer_channels(in_channels, input_size)
        self.out_channels = [stage_channels[i] for i in self.out_indices]
        self.num_features = stage_channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def _infer_channels(self, in_channels: int, input_size: int) -> List[int]:
        was_training = self.training
        self.eval()
        x = torch.zeros(1, in_channels, input_size, input_size)
        channels: List[int] = []
        for stage in self.stages:
            x = stage(x)
            if x.dim() != 4:
                raise ValueError(f"每个 stage 应输出 4D 特征图，实际 {x.dim()}D。")
            channels.append(int(x.shape[1]))
        if was_training:
            self.train()
        return channels

    def forward(self, x: torch.Tensor, **kwargs):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        if self.features_only:
            return [feats[i] for i in self.out_indices]
        return torch.flatten(self.global_pool(feats[-1]), 1)


__all__ = ["FeatureBackboneBase"]
