# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: Anchor 工具（通用几何工具，detectors 与 losses 平级共用）。
"""Anchor 工具（通用几何工具，detectors 与 losses 平级共用）。"""

import math
from typing import List, Tuple

import torch


class AnchorGenerator:
    """逐层网格 anchor 生成器（xyxy 绝对坐标）。"""

    def __init__(
        self,
        strides: Tuple[int, ...] = (8, 16, 32, 64, 128),
        base_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        scales: Tuple[float, ...] = (1.0, 1.26, 1.587),
    ):
        assert len(strides) == len(base_sizes), "strides 与 base_sizes 长度必须一致"
        self.strides = list(strides)
        self.ratios = list(ratios)
        self.scales = list(scales)
        self.base_anchors = [self._gen_base(base) for base in base_sizes]

    def _gen_base(self, base_size: float) -> torch.Tensor:
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                ws = base_size * scale * math.sqrt(ratio)
                hs = base_size * scale / math.sqrt(ratio)
                anchors.append([-ws / 2, -hs / 2, ws / 2, hs / 2])
        return torch.tensor(anchors, dtype=torch.float32)  # (A, 4)

    @property
    def num_anchors(self) -> int:
        """返回该锚框生成器的锚框数量。"""
        return len(self.ratios) * len(self.scales)

    def grid_anchors(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """由特征图形状生成各层 anchors（每层 (H*W*A, 4)）。"""
        assert len(feature_maps) == len(self.strides)
        outs = []
        for feat, stride, base in zip(feature_maps, self.strides, self.base_anchors):
            _, _, h, w = feat.shape
            shifts_x = (torch.arange(w, device=feat.device) + 0.5) * stride
            shifts_y = (torch.arange(h, device=feat.device) + 0.5) * stride
            shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing="xy")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(-1, 4)
            base = base.to(feat.device)
            anchors = (base.unsqueeze(0) + shifts.unsqueeze(1)).reshape(-1, 4)
            outs.append(anchors)
        return outs


def bbox2delta(proposals: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    """xyxy -> 编码 (dx, dy, dw, dh)。"""
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    gx = (gts[:, 0] + gts[:, 2]) / 2
    gy = (gts[:, 1] + gts[:, 3]) / 2
    gw = (gts[:, 2] - gts[:, 0]).clamp(min=1e-6)
    gh = (gts[:, 3] - gts[:, 1]).clamp(min=1e-6)
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    return torch.stack([dx, dy, dw, dh], dim=1)


def delta2bbox(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """编码 (dx,dy,dw,dh) + anchor -> xyxy。"""
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    x = ax + deltas[:, 0] * aw
    y = ay + deltas[:, 1] * ah
    w = aw * torch.exp(deltas[:, 2])
    h = ah * torch.exp(deltas[:, 3])
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


__all__ = ["AnchorGenerator", "bbox2delta", "delta2bbox"]
