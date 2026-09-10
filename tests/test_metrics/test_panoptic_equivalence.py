# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: PanopticQuality 与朴素参考实现的一致性 + 高分辨率内存/耗时回归。
"""PanopticQuality 一致性测试（CodePlan 7.7 步骤 5-3）。

v1.0.0 的 ``update`` 用 ``(p_flat[:, None, :] & g_flat[None, :, :])`` 计算交集，
会实体化 ``(P, G, H*W)`` 的布尔张量：1024×1024 图像、每类 20 个实例
就是约 4 亿个布尔值（~400MB）。修复后用「包围盒粗筛 + 交集区域局部逻辑与」。

优化必须**不改变任何数值**，因此这里把 v1.0.0 的公式原样保留为参考实现逐项对拍。
"""

import numpy as np
import pytest
import torch

from dvisionix.metrics import PanopticQuality

ID_SCALE = 1000
NUM_CATEGORIES = 4


def _reference_pq(pred: np.ndarray, gt: np.ndarray, num_categories: int = NUM_CATEGORIES):
    """v1.0.0 的原始实现（朴素全对全广播），作为数值基准。"""
    pred_cat = pred // ID_SCALE
    gt_cat = gt // ID_SCALE
    categories = set(np.unique(pred_cat)) | set(np.unique(gt_cat))
    categories = {c for c in categories if 0 <= c < num_categories}

    tp_iou, tp, fp, fn = {}, {}, {}, {}
    for cat in categories:
        pm = pred[pred_cat == cat]
        gm = gt[gt_cat == cat]
        p_masks = [pred == i for i in np.unique(pm)]
        g_masks = [gt == j for j in np.unique(gm)]
        if not p_masks or not g_masks:
            fp[cat] = fp.get(cat, 0) + len(p_masks)
            fn[cat] = fn.get(cat, 0) + len(g_masks)
            continue
        p_flat = np.stack([m.reshape(-1) for m in p_masks])
        g_flat = np.stack([m.reshape(-1) for m in g_masks])
        inter = (p_flat[:, None, :] & g_flat[None, :, :]).sum(axis=2)
        union = (p_flat[:, None, :].sum(axis=2) + g_flat[None, :, :].sum(axis=2) - inter).clip(
            min=1
        )
        iou = inter / union

        matched = set()
        total = 0.0
        for pi in range(len(p_masks)):
            cands = [gi for gi in range(len(g_masks)) if gi not in matched and iou[pi, gi] >= 0.5]
            if not cands:
                continue
            gi = max(cands, key=lambda g: iou[pi, g])
            matched.add(gi)
            total += float(iou[pi, gi])
        tp[cat] = tp.get(cat, 0) + len(matched)
        fp[cat] = fp.get(cat, 0) + len(p_masks) - len(matched)
        fn[cat] = fn.get(cat, 0) + len(g_masks) - len(matched)
        tp_iou[cat] = tp_iou.get(cat, 0.0) + total

    cats = [
        c for c in set(tp) | set(fp) | set(fn) if (tp.get(c, 0) + fp.get(c, 0) + fn.get(c, 0)) > 0
    ]
    if not cats:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0}
    pq_sum = rq_sum = sq_num = sq_den = 0.0
    for c in cats:
        t, f, n = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        denom = t + 0.5 * f + 0.5 * n
        pq_sum += tp_iou.get(c, 0.0) / denom if denom > 0 else 0.0
        rq_sum += t / denom if denom > 0 else 0.0
        if t > 0:
            sq_num += tp_iou.get(c, 0.0) / t
            sq_den += 1.0
    return {
        "PQ": float(pq_sum / len(cats)),
        "SQ": float(sq_num / sq_den) if sq_den > 0 else 0.0,
        "RQ": float(rq_sum / len(cats)),
    }


def _random_panoptic(rng, size, n_instances, categories=(1, 2)):
    """随机放置若干矩形实例，返回 (pred, gt) 的 id 图。"""

    def build(shift):
        ids = np.zeros((size, size), dtype=np.int64)
        for k in range(n_instances):
            cat = categories[k % len(categories)]
            h = int(rng.integers(size // 8, size // 3))
            w = int(rng.integers(size // 8, size // 3))
            y0 = int(rng.integers(0, max(1, size - h)))
            x0 = int(rng.integers(0, max(1, size - w)))
            y1 = min(y0 + h + shift, size)
            x1 = min(x0 + w + shift, size)
            ids[y0:y1, x0:x1] = cat * ID_SCALE + (k + 1)
        return ids

    return build(2), build(0)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_naive_reference(seed):
    rng = np.random.default_rng(seed)
    pred, gt = _random_panoptic(rng, size=64, n_instances=6)

    metric = PanopticQuality(num_categories=NUM_CATEGORIES, id_scale=ID_SCALE)
    metric.update(torch.from_numpy(pred), torch.from_numpy(gt))
    got = metric.compute()
    expected = _reference_pq(pred, gt)

    for key in ("PQ", "SQ", "RQ"):
        assert got[key] == pytest.approx(expected[key], abs=1e-12), f"{key} 与参考实现不一致"


@pytest.mark.unit
def test_empty_prediction_matches_reference():
    rng = np.random.default_rng(7)
    _, gt = _random_panoptic(rng, size=48, n_instances=4)
    pred = np.zeros_like(gt)

    metric = PanopticQuality(num_categories=NUM_CATEGORIES, id_scale=ID_SCALE)
    metric.update(torch.from_numpy(pred), torch.from_numpy(gt))
    assert metric.compute() == pytest.approx(_reference_pq(pred, gt), abs=1e-12)


@pytest.mark.unit
def test_bbox_prefilter_does_not_change_iou():
    """包围盒不相交的实例对 IoU 必须仍是 0（粗筛不能漏掉真交集）。"""
    metric = PanopticQuality(num_categories=NUM_CATEGORIES, id_scale=ID_SCALE)
    disjoint_a = np.zeros((16, 16), dtype=bool)
    disjoint_b = np.zeros((16, 16), dtype=bool)
    disjoint_a[0:4, 0:4] = True
    disjoint_b[8:12, 8:12] = True
    iou = metric._iou_matrix([disjoint_a], [disjoint_b])
    assert iou[0, 0] == 0.0

    overlapping = np.zeros((16, 16), dtype=bool)
    overlapping[2:6, 2:6] = True
    iou = metric._iou_matrix([disjoint_a], [overlapping])
    # 交集 2x2=4，并集 16+16-4=28
    assert iou[0, 0] == pytest.approx(4 / 28)


@pytest.mark.integration
def test_high_resolution_is_fast_and_bounded():
    """高分辨率回归：v1.0.0 在 1024×1024 上会实体化 ~400MB 的中间张量。"""
    import time

    size, grid = 1024, 6  # 36 个实例
    ids = np.zeros((size, size), dtype=np.int64)
    step = size // grid
    for gy in range(grid):
        for gx in range(grid):
            ids[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step] = ID_SCALE + (
                gy * grid + gx + 1
            )

    metric = PanopticQuality(num_categories=3, id_scale=ID_SCALE)
    start = time.perf_counter()
    metric.update(torch.from_numpy(ids), torch.from_numpy(ids))
    elapsed = time.perf_counter() - start
    result = metric.compute()

    # 预测与 GT 完全相同时应全部匹配上：PQ = SQ = RQ = 1
    assert result["PQ"] == pytest.approx(1.0, abs=1e-6)
    assert result["SQ"] == pytest.approx(1.0, abs=1e-6)
    assert result["RQ"] == pytest.approx(1.0, abs=1e-6)
    assert elapsed < 20.0, f"1024×1024 / 36 实例耗时 {elapsed:.1f}s，疑似退回全对全广播"
