# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Swin-UNet 完整装配（Swin encoder + SwinUNetDecoder 跳连）。
"""Swin-UNet 完整装配（Swin encoder + SwinUNetDecoder 跳连）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ...registry import BACKBONES, HEADS, MODELS
from ..base import BaseModel


@MODELS.register()
@MODELS.register(name="swin_unet")
class SwinUNet(BaseModel):
    """Swin-UNet 完整装配：SwinBackbone（多尺度 encoder）+ SwinUNetDecoder（PatchExpand + 跳连）。"""

    def __init__(
        self,
        backbone: Dict[str, Any],
        num_classes: int,
        decoder: Optional[Dict[str, Any]] = None,
        d_model: int = 64,
        upsample: bool = True,
        **kwargs,
    ):
        super().__init__(task_type="segmentation")
        bb_cfg = dict(backbone)
        bb_cfg.setdefault("features_only", True)
        self.backbone = BACKBONES.build(bb_cfg)
        dec_cfg = dict(decoder) if decoder else {"type": "swin_unet_decoder"}
        dec_cfg.setdefault("in_channels_list", list(self.backbone.out_channels))
        dec_cfg.setdefault("num_classes", num_classes)
        dec_cfg.setdefault("d_model", d_model)
        self.decoder = HEADS.build(dec_cfg)
        self.num_classes = getattr(self.decoder, "num_classes", num_classes)
        self.upsample = upsample

    def forward(self, x: torch.Tensor, **kwargs):
        """SwinUNet 前向：x -> 分割 logits (B, num_classes, H, W)。"""
        out = self.decoder(self.backbone(x))
        if self.upsample and out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


__all__ = ["SwinUNet"]
