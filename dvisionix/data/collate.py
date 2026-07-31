# -*- coding: utf-8 -*-
"""数据整理（collate）函数。

- 分类任务：默认 PyTorch collate 即可。
- 检测任务：boxes/labels 是变长，需要手动堆叠 image、保留 list。
- 分割任务：mask 同尺寸，可直接 stack。
"""

from typing import Any, Dict, List

import torch


def detection_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检测任务的 collate：image 堆叠为 (B, C, H, W)，boxes/labels 保留 list。"""
    images = torch.stack([b["image"] for b in batch], dim=0)
    out: Dict[str, Any] = {
        "image": images,
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
    }
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


def segmentation_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分割任务的 collate：image 与 mask 一起 stack（mask 保持 long）。"""
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    out: Dict[str, Any] = {"image": images, "mask": masks}
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


__all__ = ["detection_collate", "segmentation_collate"]