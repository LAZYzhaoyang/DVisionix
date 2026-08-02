# -*- coding: utf-8 -*-
"""分割组合模型：SegmentationModel（backbone + neck(可选) + 分割头）。"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .base import BaseModel
from ..registry import MODELS, BACKBONES, NECKS, HEADS

_UNET_HEAD_TYPES = ("unet_decoder", "UNetDecoder")


@MODELS.register()
@MODELS.register(name="segmentation_model")
class SegmentationModel(BaseModel):
    """骨干 + 分割头组合模型（可选 neck）。

    配置示例::

        model:
          type: segmentation_model
          num_classes: 21
          backbone: {type: timm_backbone, name: resnet18, features_only: true, out_indices: [1,2,3,4]}
          head: {type: deeplabv3_head, num_classes: 21}

    分割头支持：seg_head（1x1）/ fcn_head / deeplabv3_head / unet_decoder（需多尺度特征）。
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        out_indices: Optional[list] = None,
        num_classes: Optional[int] = None,
        upsample: bool = True,
        **kwargs,
    ):
        super().__init__(task_type="segmentation")
        bb_cfg = dict(backbone)
        bb_cfg.setdefault("features_only", True)
        if out_indices is not None:
            bb_cfg["out_indices"] = out_indices
        self.backbone = BACKBONES.build(bb_cfg)

        head_type = head.get("type") if isinstance(head, dict) else str(head)
        if neck is not None:
            neck_cfg = dict(neck)
            neck_cfg.setdefault("in_channels", self.backbone.out_channels)
            self.neck = NECKS.build(neck_cfg)
            neck_out = getattr(self.neck, "out_channels", None)
            self.in_channels = (
                neck_out if isinstance(neck_out, int)
                else (neck_out[-1] if isinstance(neck_out, (list, tuple)) else self.backbone.out_channels[-1])
            )
        else:
            self.neck = None
            self.in_channels = self.backbone.out_channels[-1]

        head_cfg = dict(head)
        if head_type in _UNET_HEAD_TYPES:
            head_cfg.setdefault("in_channels_list", list(self.backbone.out_channels))
        else:
            head_cfg.setdefault("in_channels", self.in_channels)
        if "num_classes" not in head_cfg and num_classes is not None:
            head_cfg["num_classes"] = num_classes
        self.head = HEADS.build(head_cfg)
        self.num_classes = getattr(self.head, "num_classes", num_classes)
        self.upsample = upsample

    def extract_features(self, x: torch.Tensor):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feats = self.extract_features(x)
        if isinstance(self.head, HEADS.get("unet_decoder")) or type(self.head).__name__ == "UNetDecoder":
            out = self.head(feats)
        else:
            feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
            out = self.head(feat)
        if self.upsample and out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


__all__ = ["SegmentationModel"]