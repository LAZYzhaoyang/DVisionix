# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 超参搜索工具：对配置参数网格/随机采样批量跑 train.py 并汇总最优指标。
"""超参搜索工具：对配置参数网格/随机采样批量跑 train.py 并汇总最优指标。

用法::

    python tools/hparam_search.py \
        --config configs/classification/demo_synthetic.yaml \
        --param-spec hparam_search.yaml \
        --num-trials 8                # 指定则随机采样；缺省为笛卡尔积全组合
        --monitor val_accuracy --mode max \
        --max-jobs 2
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dvisionix.utils import get_logger  # noqa: E402


def load_param_spec(path):
    """加载参数网格 YAML（点路径 -> 候选值列表）。"""
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    assert isinstance(spec, dict) and spec, "param-spec 需为 {点路径: [候选值,...]} 字典"
    return spec


def combinations(spec):
    """对参数网格做笛卡尔积，返回 (keys, combos)。"""
    keys = list(spec.keys())
    values = list(itertools.product(*[spec[k] for k in keys]))
    return keys, values


def random_trials(spec, num_trials, rng):
    """随机采样 num_trials 组参数组合。"""
    keys = list(spec.keys())
    return keys, [tuple(rng.choice(spec[k]) for k in keys) for _ in range(num_trials)]


def format_value(v):
    """把参数值格式化为 CLI 字符串（列表转 [a,b]）。"""
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def to_cfg_options(keys, combo):
    """Convert (keys, combo) into a list of ``k=v`` strings for --cfg-options."""
    return [f"{k}={format_value(v)}" for k, v in zip(keys, combo)]


def best_from_history(work_dir, monitor, mode):
    """从 work_dir/history.csv 读取监控指标的最优值。"""
    path = os.path.join(work_dir, "history.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = None
    for r in rows:
        if monitor not in r:
            continue
        try:
            val = float(r[monitor])
        except (TypeError, ValueError):
            continue
        if best is None or (val > best[1] if mode == "max" else val < best[1]):
            best = (r, val)
    return best


def main():
    """超参搜索入口：组合/采样 -> 批量跑 train.py -> 汇总 search_results.csv。"""
    parser = argparse.ArgumentParser(description="DVisionix 超参搜索")
    parser.add_argument("--config", required=True, help="基础配置文件")
    parser.add_argument("--param-spec", required=True, help="参数网格 YAML（点路径 -> 候选值列表）")
    parser.add_argument("--num-trials", type=int, default=None, help="随机采样次数（缺省笛卡尔积）")
    parser.add_argument("--monitor", default="val_loss", help="汇总监控指标")
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument("--max-jobs", type=int, default=1, help="并行 job 数")
    parser.add_argument("--seed", type=int, default=0, help="随机采样种子")
    parser.add_argument(
        "--work-root", default=None, help="搜索实验根目录（缺省为 base 配置的 work_dir 下 search/）"
    )
    args = parser.parse_args()

    logger = get_logger("hparam_search")
    base_work = args.work_root or "runs"
    base_work = os.path.join(base_work, "search")
    os.makedirs(base_work, exist_ok=True)

    spec = load_param_spec(args.param_spec)
    if args.num_trials:
        import numpy as np

        keys, combos = random_trials(spec, args.num_trials, np.random.default_rng(args.seed))
    else:
        keys, combos = combinations(spec)
    logger.info(f"共 {len(combos)} 组实验")

    processes = []
    for i, combo in enumerate(combos):
        work = os.path.join(base_work, f"trial_{i:03d}")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
            "--config",
            args.config,
            "--cfg-options",
            *to_cfg_options(keys, combo),
            "--work-dir",
            work,
        ]
        processes.append((i, combo, work, cmd))

    # 按 max_jobs 分批串行提交（每个 job 独立进程，实验间完全隔离）
    idx = 0
    while idx < len(processes):
        batch = processes[idx : idx + args.max_jobs]
        for job in batch:
            _run(job, logger)
        idx += args.max_jobs

    # 汇总
    out_path = os.path.join(base_work, "search_results.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(keys) + [args.monitor, "work_dir"])
        for i, combo, work, _ in processes:
            best = best_from_history(work, args.monitor, args.mode)
            row = list(combo)
            if best is not None:
                row += [best[1], work]
            else:
                row += ["", work]
            writer.writerow(row)
    logger.info(f"搜索结果已写入 {out_path}")


def _run(job, logger):
    _, combo, work, cmd = job
    shown = cmd[6:]
    if "--work-dir" in shown:
        shown = shown[: shown.index("--work-dir")]
    logger.info("run %s: %s", work, " ".join(shown))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
