# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 几何同步变换。
"""几何同步变换。

任务无关，但会同步处理 ``boxes`` (xyxy) 与 ``mask``（如果存在），
因此分类 / 检测 / 分割都可以复用同一套。

约定：boxes 是 numpy float32 (N, 4) [x1, y1, x2, y2] 绝对坐标；
mask 是 numpy int64 (H, W)。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ...registry import TRANSFORMS
from ..sample import Sample
from .base import BaseTransform


def _validate_boxes_labels(
    boxes: Optional[np.ndarray], labels: Optional[np.ndarray], op: str
) -> None:
    """校验 boxes 与 labels 数量一致。

    几何变换靠「同一个布尔掩码同时裁剪 boxes 和 labels」维持二者的对应关系；
    一旦长度不一致，任何裁剪都会让标签与框静默错位。因此必须在入口就失败，
    而不是产出「看起来正常」的错误样本（CodePlan 7.4 步骤 2-3）。
    """
    if boxes is None or labels is None:
        return
    if len(boxes) != len(labels):
        raise ValueError(
            f"{op}: boxes 与 labels 数量不一致（boxes={len(boxes)}, labels={len(labels)}），"
            f"几何变换会破坏二者的对应关系"
        )


def _filter_invalid_boxes(boxes: np.ndarray, labels: Optional[np.ndarray]):
    """过滤退化的 box（w<=0 或 h<=0），并同步裁剪 labels。"""
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if labels is not None:
        if len(labels) != len(valid):
            raise ValueError(
                f"boxes 与 labels 数量不一致（boxes={len(valid)}, labels={len(labels)}），"
                f"无法同步过滤"
            )
        labels = labels[valid]
    return boxes, labels


def _ensure_crop_size(sample: Sample, size: Tuple[int, int], on_small: str, op: str) -> None:
    """确保 image/mask 不小于裁剪尺寸，按 ``on_small`` 就地调整 sample。

    Args:
        sample: 待处理的 Sample（就地修改 ``image`` / ``mask``）。
        size: 目标裁剪尺寸 ``(th, tw)``。
        on_small: ``error``（默认）尺寸不足直接报错；``pad`` 右下补零（boxes 坐标
            仍在原图范围内，因此依然有效）；``resize`` 会被拒绝 —— 缩放会改变
            boxes 坐标，几何管线应改用 ``BoxSyncResize`` 显式放大。
        op: 调用方名称，用于错误信息定位。
    """
    img = sample["image"]
    h, w = img.shape[:2]
    th, tw = size
    if h >= th and w >= tw:
        return
    if on_small == "error":
        raise ValueError(
            f"{op}: 输入尺寸 {h}x{w} 小于目标裁剪尺寸 {th}x{tw}。"
            f"请先用 BoxSyncResize 放大，或显式设置 on_small='pad'。"
        )
    if on_small == "pad":
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        sample["image"] = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if "mask" in sample:
            sample["mask"] = np.pad(sample["mask"], ((0, pad_h), (0, pad_w)), constant_values=0)
        return
    if on_small == "resize":
        raise ValueError(
            f"{op}: 不支持 on_small='resize' —— 缩放会改变 boxes 坐标，"
            f"请改用 BoxSyncResize 放大图像，或用 on_small='pad'。"
        )
    raise ValueError(f"{op}: on_small 必须是 'error' / 'pad' / 'resize'，当前 {on_small!r}")


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_resize")
class BoxSyncResize(BaseTransform):
    """resize image + 同步缩放 boxes（xyxy）+ mask。"""

    name = "box_sync_resize"

    def __init__(self, size: Optional[Tuple[int, int]] = None, max_size: Optional[int] = None):
        if size is None and max_size is None:
            raise ValueError("Either size or max_size must be specified.")
        self.size, self.max_size = size, max_size

    def __call__(self, sample: Sample) -> Sample:
        _validate_boxes_labels(sample.get("boxes"), sample.get("labels"), "BoxSyncResize")
        img = sample["image"]
        h, w = img.shape[:2]
        if self.size is not None:
            th, tw = self.size
            sh, sw = th / h, tw / w
        else:
            s = self.max_size / max(h, w)
            sh = sw = s
            th, tw = int(round(h * s)), int(round(w * s))
        sample["image"] = cv2.resize(img, (tw, th))

        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] *= sw
            boxes[:, [1, 3]] *= sh
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th)
            labels = sample.get("labels")
            boxes, labels = _filter_invalid_boxes(boxes, labels)
            sample["boxes"] = boxes
            if labels is not None:
                sample["labels"] = labels

        if "mask" in sample:
            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.int64), (tw, th), interpolation=cv2.INTER_NEAREST
            )

        meta = sample.get("meta") or {}
        meta["scale"] = (sh, sw)
        meta["original_size"] = (h, w)
        sample["meta"] = meta
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_random_hflip")
class BoxSyncRandomHorizontalFlip(BaseTransform):
    """随机水平翻转 image + 同步翻转 boxes。"""

    name = "box_sync_random_hflip"

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        _validate_boxes_labels(
            sample.get("boxes"), sample.get("labels"), "BoxSyncRandomHorizontalFlip"
        )
        if np.random.random() >= self.p:
            return sample
        img = sample["image"]
        h, w = img.shape[:2]
        sample["image"] = cv2.flip(img, 1)
        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            sample["boxes"] = boxes
        if "mask" in sample:
            sample["mask"] = cv2.flip(sample["mask"], 1)
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_random_crop")
class BoxSyncRandomCrop(BaseTransform):
    """随机裁剪 image/mask，同步平移 boxes 并丢弃越界后为空的 box。

    Args:
        size: 目标 ``(H, W)``。
        on_small: 输入小于目标尺寸时的行为，见 ``_ensure_crop_size``。
            v1.0.0 为静默返回原图（裁剪完全失效），现默认 ``error``。

    注意：要真正产生随机偏移，输入必须先大于目标尺寸 —— 检测预置管线
    因此先用 ``BoxSyncResize`` 放大到 1.1 倍再裁剪（CodePlan 7.1.2 D9）。
    """

    name = "box_sync_random_crop"

    def __init__(self, size: Tuple[int, int], on_small: str = "error"):
        self.size = size
        self.on_small = on_small

    def __call__(self, sample: Sample) -> Sample:
        _validate_boxes_labels(sample.get("boxes"), sample.get("labels"), "BoxSyncRandomCrop")
        _ensure_crop_size(sample, self.size, self.on_small, "BoxSyncRandomCrop")
        img = sample["image"]
        h, w = img.shape[:2]
        th, tw = self.size
        y = np.random.randint(0, h - th + 1)
        x = np.random.randint(0, w - tw + 1)
        sample["image"] = img[y : y + th, x : x + tw]
        if "mask" in sample:
            sample["mask"] = sample["mask"][y : y + th, x : x + tw]
        if "boxes" in sample and len(sample["boxes"]) > 0:
            boxes = sample["boxes"].astype(np.float32).copy()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, tw)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, th)
            labels = sample.get("labels")
            boxes, labels = _filter_invalid_boxes(boxes, labels)
            sample["boxes"] = boxes
            if labels is not None:
                sample["labels"] = labels
        return sample


@TRANSFORMS.register()
@TRANSFORMS.register(name="box_sync_pad")
class BoxSyncPad(BaseTransform):
    """将 image pad 到目标尺寸（右下 pad），boxes 坐标不变。"""

    name = "box_sync_pad"

    def __init__(self, size: Tuple[int, int], pad_value: int = 0, mask_value: int = 0):
        self.size = size
        self.pad_value = pad_value
        self.mask_value = mask_value

    def __call__(self, sample: Sample) -> Sample:
        img = sample["image"]
        h, w = img.shape[:2]
        th, tw = self.size
        if h >= th and w >= tw:
            return sample
        pad_h, pad_w = max(0, th - h), max(0, tw - w)
        sample["image"] = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.pad_value
        )
        if "mask" in sample:
            sample["mask"] = np.pad(
                sample["mask"], ((0, pad_h), (0, pad_w)), constant_values=self.mask_value
            )
        return sample
