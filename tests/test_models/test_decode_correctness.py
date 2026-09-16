# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 检测解码契约回归：展平索引必须还原出正确的 (类别, 位置) 组合。
"""检测解码契约回归测试。

背景（为什么需要这个文件）：

``scores`` 由 ``(N, C)`` 行主序展平而来，因此索引含义是 ``idx = loc * C + cls``：
类别要**取模**、位置要**整除**。v1.1.0 的 YOLO / FCOS / RetinaNet 解码三处都误用了
"位置数"当除数：

    labels  = top_idx // n_loc        # 应为 % num_classes
    loc_idx = top_idx %  n_loc        # 应为 // num_classes

后果是**标签与框同时错位**（实测：应得 label=1 却得 2，框中心从 (6,6) 跑到 (2,6)），
所有检测指标（mAP/mAP_50/mAP_75）与推理/导出结果都不可信；
而**训练 loss 完全正常**（损失端语义正确），所以整个 CI 都没有任何信号。

原有的 ``test_detectors.py`` 只断言 ``boxes.shape[1] == 4`` 与三者长度相等，
标签落在 ``[0, C)`` 内也不会越界，因此这类错误可以长期存在。
本文件用「已知类别 + 已知位置」的受控输入把 (label, box) 对应关系钉死。
"""

import torch

from dvisionix.models.detectors.fcos import fcos_decode
from dvisionix.models.detectors.retinanet import retinanet_decode
from dvisionix.models.detectors.yolo import yolo_decode
from dvisionix.models.layers.anchors import AnchorGenerator


def _pick(cls_chw: torch.Tensor, cls_id: int, loc_id: int, value: float = 20.0) -> None:
    """在 (C, h, w) 的 cls 上把「类别 cls_id、空间位置 loc_id」置为高分。

    解码端把分数展平成 ``(N, C).reshape(-1)``，故该点的索引是
    ``idx = loc_id * num_classes + cls_id`` —— 正是标签取模、位置取整要还原的目标。
    """
    cls_chw[cls_id].view(-1)[loc_id] = value


def test_yolo_decode_label_and_location_match():
    """YOLO：高分位置必须解出 (cls_id, 对应位置中心)。"""
    num_classes, h, w, stride = 3, 2, 2, 4
    cls = torch.full((1, num_classes, h, w), -20.0)
    reg = torch.zeros((1, 4, h, w))  # ltrb=0 → 框退化为位置中心点，便于断言坐标
    cls_id, loc_id = 1, 2  # loc 2 → 中心 (2, 6)
    _pick(cls[0], cls_id, loc_id)

    boxes, _scores, labels = yolo_decode(
        {"cls": [cls], "reg": [reg]}, (8, 8), strides=[stride], score_threshold=0.5
    )
    assert boxes[0].shape[0] == 1, "只应保留一个高分框"
    assert int(labels[0][0]) == cls_id, f"类别错位：期望 {cls_id}，实得 {int(labels[0][0])}"
    cx = float((boxes[0][0, 0] + boxes[0][0, 2]) / 2)
    cy = float((boxes[0][0, 1] + boxes[0][0, 3]) / 2)
    assert (cx, cy) == (2.0, 6.0), f"框位置错位：中心期望 (2.0, 6.0)，实得 ({cx}, {cy})"


def test_fcos_decode_label_and_location_match():
    """FCOS：同 YOLO，且 centerness 通道不参与类别索引。"""
    num_classes, h, w, stride = 3, 2, 2, 2
    cls = torch.full((1, num_classes, h, w), -20.0)
    reg = torch.zeros((1, 4, h, w))
    center = torch.full((1, 1, h, w), 20.0)  # centerness 恒为 1，score = cls_prob
    cls_id, loc_id = 1, 3  # loc 3 → 中心 (3, 3)；图取 64 以免 clamp 改变中心
    _pick(cls[0], cls_id, loc_id)

    boxes, _scores, labels = fcos_decode(
        {"cls": [cls], "reg": [reg], "center": [center]},
        (64, 64),
        strides=[stride],
        score_threshold=0.5,
    )
    assert boxes[0].shape[0] == 1
    assert int(labels[0][0]) == cls_id, f"类别错位：期望 {cls_id}，实得 {int(labels[0][0])}"
    cx = float((boxes[0][0, 0] + boxes[0][0, 2]) / 2)
    cy = float((boxes[0][0, 1] + boxes[0][0, 3]) / 2)
    assert (cx, cy) == (3.0, 3.0), f"框位置错位：中心期望 (3.0, 3.0)，实得 ({cx}, {cy})"


def test_retinanet_decode_label_and_location_match():
    """RetinaNet：展平维度是 (n_loc*A, C)，除数同样必须是 num_classes。"""
    num_classes, h, w, stride = 3, 2, 2, 2
    # base_size=2 让 anchor 不越过图像左/上边界（否则会被 clamp 改变中心）
    anchor_gen = AnchorGenerator(strides=(stride,), base_sizes=(2,), ratios=(1.0,), scales=(1.0,))
    anchors_per_level = anchor_gen.grid_anchors([torch.zeros(1, 1, h, w)])

    cls = torch.full((1, num_classes, h, w), -20.0)
    reg = torch.zeros((1, 4, h, w))
    cls_id, loc_id = 1, 3
    _pick(cls[0], cls_id, loc_id)

    boxes, _scores, labels = retinanet_decode(
        {"cls": [cls], "reg": [reg]}, (64, 64), anchor_gen, score_threshold=0.5
    )
    assert boxes[0].shape[0] == 1
    assert int(labels[0][0]) == cls_id, f"类别错位：期望 {cls_id}，实得 {int(labels[0][0])}"
    # anchor 是单尺度单比例，中心即网格中心
    cx = float((boxes[0][0, 0] + boxes[0][0, 2]) / 2)
    cy = float((boxes[0][0, 1] + boxes[0][0, 3]) / 2)
    assert (cx, cy) == (3.0, 3.0), f"框位置错位：中心期望 (3.0, 3.0)，实得 ({cx}, {cy})"
    assert anchors_per_level[0].shape[0] == h * w


def test_decode_labels_never_exceed_num_classes():
    """随机高分下，标签必须始终落在 [0, num_classes)（旧实现会越界到位置数）。"""
    torch.manual_seed(0)
    num_classes, h, w, stride = 5, 4, 4, 8
    cls = torch.randn(2, num_classes, h, w)
    reg = torch.zeros((2, 4, h, w))
    boxes, _scores, labels = yolo_decode(
        {"cls": [cls], "reg": [reg]}, (32, 32), strides=[stride], score_threshold=0.0
    )
    for lb in labels:
        if lb.numel():
            assert int(lb.max()) < num_classes, "标签越界：说明索引仍按位置数取整"
            assert int(lb.min()) >= 0
