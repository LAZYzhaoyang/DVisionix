# -*- coding: utf-8 -*-
"""SegFormer 变体分割头（overlap patch embed + MixFFN，compact）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel
from ...layers import LayerNorm2d, MixFFN


@HEADS.register()
@HEADS.register(name="segformer_v2_head")
class SegFormerV2Head(BaseModel):
    """SegFormer 变体解码头：每层 overlap patch embed（4x4 stride2）下采样后过 N 个
    MixFFN 块（LN + 3x3 深度卷积 + MLP），多尺度融合后 MLP 解码。

    多尺度输入（input_style="multi_scale"），增强特征提取，配合任意骨干即插即用。
    """

    input_style = "multi_scale"  # 多尺度输入（装配器注入 in_channels_list）

    def __init__(
        self,
        in_channels_list,
        num_classes,
        d_model: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(task_type="segmentation")
        self.num_classes = num_classes
        self.d_model = d_model
        self.in_channels_list = list(in_channels_list)
        levels = len(in_channels_list)

        self.overlap = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for c in in_channels_list:
            self.overlap.append(
                nn.Sequential(
                    nn.Conv2d(c, d_model, kernel_size=4, stride=2, padding=1),
                    LayerNorm2d(d_model),
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                )
            )
            self.blocks.append(nn.ModuleList([MixFFN(d_model) for _ in range(num_blocks)]))
        self.decode = nn.Sequential(
            nn.Conv2d(d_model * levels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_model, num_classes, 1),
        )

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        outs = []
        for i, f in enumerate(feats):
            x = self.overlap[i](f)
            for blk in self.blocks[i]:
                x = blk(x)
            outs.append(x)
        target = outs[0].shape[-2:]
        fused = torch.cat(
            [F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs],
            dim=1,
        )
        return self.decode(fused)


__all__ = ["SegFormerV2Head"]
