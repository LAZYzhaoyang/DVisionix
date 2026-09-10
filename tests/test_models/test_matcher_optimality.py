# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 匈牙利匹配的最优性与形状契约测试（含 300×30 的性能回归）。
"""匈牙利匹配的最优性与形状契约测试（CodePlan 7.7 步骤 5-3）。

v1.0.0 的 ``_hungarian`` 无论输入形状如何都把代价矩阵补齐成
``max(n, m) × max(n, m)`` 的方阵，再跑纯 Python 三重循环的 O(n²m) 算法。
DETR 的典型形状是 ``300 queries × 30 GT``，于是问题被放大成 300×300 ——
实测单次匹配 **13.6 秒**（见 tools/benchmark.py 的 matcher 场景）。

修复方式：该实现要求「行数 ≤ 列数」，行多于列时先转置再求解，
因此**不需要补方阵**。为了让这次改动可信，本文件同时锁定：

1. 匹配结果必须仍是**最小代价**（用穷举对拍）；
2. 返回的索引必须在转置前后都正确（行/列不能互换）；
3. 形状边界：n<m、n>m、n==m、含重复代价的退化情形。
"""

import itertools

import numpy as np
import pytest

from dvisionix.models.losses.detection.matcher import HungarianMatcher, _hungarian


def _brute_force_cost(cost: np.ndarray) -> float:
    """穷举最小匹配代价（仅用于小规模对拍）。"""
    n, m = cost.shape
    k = min(n, m)
    if n <= m:
        best = np.inf
        for cols in itertools.combinations(range(m), k):
            for perm in itertools.permutations(cols):
                best = min(best, sum(cost[i, perm[i]] for i in range(n)))
        return float(best)
    best = np.inf
    for rows in itertools.combinations(range(n), k):
        for perm in itertools.permutations(rows):
            best = min(best, sum(cost[perm[j], j] for j in range(m)))
    return float(best)


def _matching_cost(cost: np.ndarray, rows, cols) -> float:
    return float(sum(cost[r, c] for r, c in zip(rows, cols)))


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (5, 3), (1, 4), (4, 1), (2, 2), (4, 4)])
def test_hungarian_is_optimal_and_valid(shape):
    rng = np.random.default_rng(0)
    for _ in range(20):
        cost = rng.random(shape) * 10
        rows, cols = _hungarian(cost)

        k = min(shape)
        assert len(rows) == len(cols) == k, f"{shape}: 匹配数应为 {k}"
        assert len(set(rows.tolist())) == k, "行索引重复"
        assert len(set(cols.tolist())) == k, "列索引重复"
        assert all(0 <= r < shape[0] for r in rows), "行索引越界"
        assert all(0 <= c < shape[1] for c in cols), "列索引越界"

        got = _matching_cost(cost, rows, cols)
        assert got == pytest.approx(
            _brute_force_cost(cost), abs=1e-9
        ), f"{shape}: 匹配代价 {got} 不是最优"


@pytest.mark.unit
def test_hungarian_handles_duplicate_costs():
    cost = np.ones((4, 6))
    rows, cols = _hungarian(cost)
    assert sorted(rows.tolist()) == [0, 1, 2, 3]
    assert len(set(cols.tolist())) == 4


@pytest.mark.unit
def test_matcher_returns_correct_orientation():
    """转置分支最容易出的错是行列互换：用可辨识的代价矩阵验证方向。"""
    torch = pytest.importorskip("torch")

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=0.0, cost_giou=0.0)
    # Q=4 个 query、M=2 个 GT：行多于列，必然走转置分支
    # 分类代价占主导（bbox/giou 权重为 0），因此匹配由 logits 决定
    logits = torch.zeros(4, 4)  # (Q, num_classes + 1)
    logits[2, 1] = 20.0  # query2 强烈指向类别 1
    logits[0, 0] = 20.0  # query0 强烈指向类别 0
    boxes = torch.zeros(4, 4)
    gt_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]])
    gt_labels = torch.tensor([1, 0])

    rows, cols = matcher(logits, boxes, gt_boxes, gt_labels)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    assert mapping[0] == 2, f"GT0（类别 1）应匹配 query2，实际映射 {mapping}"
    assert mapping[1] == 0, f"GT1（类别 0）应匹配 query0，实际映射 {mapping}"


@pytest.mark.unit
def test_matcher_handles_empty_gt():
    torch = pytest.importorskip("torch")
    matcher = HungarianMatcher()
    rows, cols = matcher(torch.zeros(5, 3), torch.zeros(5, 4), torch.zeros(0, 4), torch.zeros(0))
    assert rows.numel() == 0 and cols.numel() == 0


@pytest.mark.integration
def test_matcher_large_shape_is_fast():
    """性能回归：300×30 的形状必须在毫秒级完成（修复前约 13.6 秒）。

    阈值取得很宽松（2 秒），只用于捕捉「又回到补方阵」这类量级退化，
    避免在慢机器上产生假阳性。
    """
    import time

    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    matcher = HungarianMatcher()
    args = (
        torch.randn(300, 6),
        torch.rand(300, 4),
        torch.rand(30, 4),
        torch.randint(0, 5, (30,)),
    )
    start = time.perf_counter()
    rows, cols = matcher(*args)
    elapsed = time.perf_counter() - start

    assert len(rows) == len(cols) == 30
    assert elapsed < 2.0, f"300x30 匹配耗时 {elapsed:.2f}s，疑似退回补方阵的 O(max(n,m)^2 * m) 实现"
