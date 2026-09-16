# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 工作目录与断点续训工具。
"""工作目录与断点续训工具。

- 工作目录默认在代码库外（``~/dvisionix_runs/<experiment>/<timestamp>``），
  可通过 ``DVISIONIX_WORK_DIR`` 环境变量或配置 ``work_dir`` 覆盖。
- ``resume`` 三态：false（全新）/ auto|latest（续最近一次任务）/ 显式路径。
"""

import hashlib
import os
from datetime import datetime
from typing import Optional

import yaml


def default_work_root() -> str:
    """默认工作根目录（代码库外）。优先级：环境变量 > 用户主目录。"""
    return os.environ.get("DVISIONIX_WORK_DIR") or os.path.join(
        os.path.expanduser("~"), "dvisionix_runs"
    )


def distributed_rank_from_env() -> Optional[int]:
    """从环境变量读取 torchrun 注入的 rank；非分布式启动时返回 None。

    torchrun 在 **任何** ``dist.init_process_group`` 之前就会注入
    ``RANK`` / ``WORLD_SIZE``，因此工作目录解析阶段就能拿到。
    """
    rank = os.environ.get("RANK")
    world = os.environ.get("WORLD_SIZE")
    if rank is None or world is None:
        return None
    try:
        if int(world) <= 1:
            return None
        return int(rank)
    except (TypeError, ValueError):
        return None


def resolve_work_dir(
    cfg,
    cli_work_dir: Optional[str] = None,
    resume: Optional[str] = None,
) -> str:
    """解析本次运行的工作目录。

    Args:
        cfg: Config（需含 experiment_name / work_dir）。
        cli_work_dir: 命令行 --work-dir。
        resume: 'auto' / 'latest' / 其它字符串 / None。

    Returns:
        工作目录绝对路径。
    """
    base = cli_work_dir or cfg.get("work_dir") or default_work_root()
    experiment = cfg.get("experiment_name", "exp")

    if resume is True:
        resume = "auto"  # `resume: true` 等同于 auto，见 find_checkpoint
    if resume in ("auto", "latest"):
        run_dir = find_latest_run(os.path.join(base, experiment))
        if run_dir is None:
            raise FileNotFoundError(
                f"No previous run found under {os.path.join(base, experiment)!r} for resume={resume}"
            )
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 配置哈希后缀：同配置可复现定位，异配置避免目录冲突（实验管理）
    run_id = f"{timestamp}-{hash_config(cfg)}"
    if distributed_rank_from_env() is not None:
        # torchrun 下各进程独立执行本函数，datetime.now() 跨秒就会得到**不同**目录：
        # rank0 的 checkpoint / 日志与其它 rank 的 history.csv 分家，`resume auto`
        # 也会选到错误（或不存在）的目录。配置哈希在所有 rank 上完全一致，
        # 因此分布式下改用确定性目录名（不带时间戳），无需任何通信。
        run_id = hash_config(cfg)
    return os.path.join(base, experiment, run_id)


def hash_config(cfg) -> str:
    """计算解析后配置的确定性哈希（sha1 前 8 位），用于实验目录标识。"""
    if hasattr(cfg, "to_dict"):
        data = cfg.to_dict()
    else:
        data = cfg
    dumped = yaml.safe_dump(data, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha1(dumped).hexdigest()[:8]


def find_latest_run(experiment_dir: str) -> Optional[str]:
    """在实验目录下找含 checkpoints/last.pt 的最新 run 目录。"""
    if not os.path.isdir(experiment_dir):
        return None
    runs = [
        os.path.join(experiment_dir, d)
        for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ]
    runs = [r for r in runs if _has_last_checkpoint(r)]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def _has_last_checkpoint(run_dir: str) -> bool:
    """run 目录下是否存在 last.pt（标准布局 checkpoints/last.pt 或嵌套子目录）。"""
    direct = os.path.join(run_dir, "checkpoints", "last.pt")
    if os.path.exists(direct):
        return True
    for root, _dirs, files in os.walk(run_dir):
        if "last.pt" in files:
            return True
    return False


def find_checkpoint(work_dir: str, resume: Optional[str]) -> Optional[str]:
    """根据 resume 配置定位检查点路径。

    - None / False -> None
    - True -> 等价于 'auto'（``resume: true`` 是极自然的写法，见下）
    - 'auto' / 'latest' -> work_dir/checkpoints/last.pt（不存在则 None）
    - 其它字符串 -> 视为显式路径
    """
    if not resume:
        return None
    if resume is True:
        # `resume: true` 会被原样当成路径返回，最终 torch.load(True) 抛
        # "AttributeError: 'bool' object has no attribute 'seek'"，
        # 而且 resolve_work_dir 还会另起一个带新时间戳的目录（既没续训又污染实验目录）。
        # 按最自然的意图当成 "auto"。
        resume = "auto"
    if not isinstance(resume, str):
        raise ValueError(
            f"resume 必须是 false / true / 'auto' / 'latest' 或 checkpoint 路径，"
            f"当前为 {type(resume).__name__}: {resume!r}"
        )
    if resume in ("auto", "latest"):
        direct = os.path.join(work_dir, "checkpoints", "last.pt")
        if os.path.exists(direct):
            return direct
        for root, _dirs, files in os.walk(work_dir):
            if "last.pt" in files:
                return os.path.join(root, "last.pt")
        return None
    return resume


def dump_config(cfg, path: str) -> None:
    """把最终解析后的配置 dump 到工作目录，保证可复现。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)


__all__ = [
    "default_work_root",
    "distributed_rank_from_env",
    "resolve_work_dir",
    "hash_config",
    "find_latest_run",
    "find_checkpoint",
    "dump_config",
]
