# -*- coding: utf-8 -*-
"""可配置的顺序骨干网络。

SequentialBackbone 允许用一组 layer 配置（stages）拼装自定义骨干网络，
无需额外写类即可接入 GeneralizedModel（backbone -> neck -> head）。

- features_only=False（默认）: 输出全局池化后的特征向量 (B, num_features)，适用于分类头。
- features_only=True: 输出多尺度特征图列表 [C1, C2, ...]，并暴露 out_channels，供 FPN / 检测头 / 分割头使用。

每个 stage 可以是单个 layer 配置字典，也可以是配置字典列表（会顺序组合为一个 stage）。
各 stage 的输出通道数通过一次 dry-run 前向自动推导，无需手工声明。
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from ...registry import BACKBONES
from ..base import BaseModel
from ..layers import build_layer

StageCfg = Union[Dict[str, Any], Sequence[Dict[str, Any]]]


@BACKBONES.register()
@BACKBONES.register(name="sequential_backbone")
class SequentialBackbone(BaseModel):
    """用 layer 配置列表拼装的顺序骨干网络。

    Args:
        stages: stage 配置列表。每个元素是一个 layer 配置字典，或配置字典列表
            （列表会被组合为一个 nn.Sequential 作为该 stage）。
        in_channels: 输入图像通道数，仅用于 dry-run 推导各 stage 输出通道。
        input_size: dry-run 使用的空间尺寸（默认 32）。
        features_only: 为 True 时 forward 返回多尺度特征图列表，否则返回全局池化特征向量。
        out_indices: features_only 模式下返回哪些 stage 的输出（None 表示返回全部 stage）。

    Examples:
        >>> cfg = [
        ...     {"type": "conv_norm_act", "in_channels": 3, "out_channels": 32, "stride": 2},
        ...     {"type": "conv_norm_act", "in_channels": 32, "out_channels": 64, "stride": 2},
        ... ]
        >>> backbone = SequentialBackbone(cfg, features_only=True)
        >>> feats = backbone(torch.randn(2, 3, 64, 64))  # List[Tensor]
        >>> backbone.out_channels  # [32, 64]
    """

    def __init__(
        self,
        stages: Sequence[StageCfg],
        in_channels: int = 3,
        input_size: int = 32,
        features_only: bool = False,
        out_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("SequentialBackbone 至少需要一个 stage 配置。")

        self.features_only = features_only
        self.stages = nn.ModuleList([self._build_stage(s) for s in stages])

        num_stages = len(self.stages)
        if out_indices is None:
            out_indices = list(range(num_stages))
        self.out_indices = [i % num_stages for i in out_indices]

        stage_channels = self._infer_channels(in_channels, input_size)
        self._all_channels = stage_channels
        self.out_channels = [stage_channels[i] for i in self.out_indices]
        self.num_features = stage_channels[-1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _build_stage(stage_cfg: StageCfg) -> nn.Module:
        if isinstance(stage_cfg, dict):
            return build_layer(stage_cfg)
        layers = [build_layer(c) for c in stage_cfg]
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _infer_channels(self, in_channels: int, input_size: int) -> List[int]:
        was_training = self.training
        self.eval()
        x = torch.zeros(1, in_channels, input_size, input_size)
        channels: List[int] = []
        for stage in self.stages:
            x = stage(x)
            if x.dim() != 4:
                raise ValueError(
                    "SequentialBackbone 要求每个 stage 输出 4D 特征图 (B, C, H, W)，"
                    f"实际得到 {x.dim()}D。"
                )
            channels.append(int(x.shape[1]))
        if was_training:
            self.train()
        return channels

    def forward(self, x, **kwargs):
        """features_only=True 返回多尺度特征图列表，否则返回全局池化特征向量。"""
        feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        if self.features_only:
            return [feats[i] for i in self.out_indices]

        pooled = self.global_pool(feats[-1])
        return torch.flatten(pooled, 1)
