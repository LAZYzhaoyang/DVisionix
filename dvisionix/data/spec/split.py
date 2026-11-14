# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集切分：SplitResult 与 build_splits（ratio / dir / list）。
"""DVX 数据集切分：``SplitResult`` 与 ``build_splits``（契约 P1-5）。

切分是**非破坏性**的（契约 P1-0）：只产出"样本下标 -> split 名"的索引映射，
绝不移动、改名或删除原始文件。

三种模式：

- ``ratio``：按 ``rel_path`` 排序后，用 ``random.Random(seed)`` 在稳定顺序上打乱，
  可按类别分层；同一输入多次调用结果完全一致。
- ``dir``：按 ``dir_map``（相对 ``spec.root``，多任务时相对每个 ``task.path``）
  判定每个样本属于哪个 split。
- ``list``：读 JSON/YAML 清单（``{split: [rel_path, ...]}``），显式指定归属。

所有模式最后都会做同一套硬校验：下标不重叠、且完整覆盖 ``[0, len(refs))``。

用法::

    from dvisionix.data.spec.schema import load_spec
    from dvisionix.data.spec.scan import scan_samples
    from dvisionix.data.spec.split import build_splits

    spec = load_spec("project/data/cls/dvx_dataset.yaml")
    refs = scan_samples(spec)
    result = build_splits(refs, spec.splits, spec)
    print(result.summary())
    result.save("work/split.json")
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from dvisionix.errors import SplitError

from .schema import SplitConfig

if TYPE_CHECKING:
    from .scan import SampleRef
    from .schema import DVXDatasetSpec

__all__ = ["SplitResult", "build_splits"]

#: 合法的 split 名字（顺序决定输出/统计的稳定顺序）
SPLIT_NAMES = ("train", "val", "test")

#: 切分模式（与 schema 保持一致）
SPLIT_MODES = ("ratio", "dir", "list")

#: ratio 之和容差（与 schema 保持一致）
RATIO_SUM_TOLERANCE = 1e-6


@dataclass
class SplitResult:
    """一次切分的结果（契约 P1-5）。

    Args:
        splits: split 名 -> 样本下标（升序；下标指向 ``build_splits`` 收到的 ``refs`` 列表）。
        stats: 统计信息，结构为 ``{total, splits: {name: {count, per_class}}, warnings}``。
        mode: 切分模式（ratio / dir / list）。
        seed: ratio 模式使用的随机种子；dir/list 模式为 None。
    """

    splits: dict[str, list[int]]
    stats: dict
    mode: str
    seed: int | None

    def save(self, path: str) -> None:
        """把切分结果保存为 JSON 文件。

        Args:
            path: 输出文件路径（父目录不存在时自动创建）。

        Raises:
            SplitError: 写入失败。
        """
        target = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(target)
        payload = {
            "splits": {name: list(indices) for name, indices in self.splits.items()},
            "stats": self.stats,
            "mode": self.mode,
            "seed": self.seed,
        }
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError as exc:
            raise SplitError(
                f"切分结果写入失败: {target}（{exc}）",
                hint="检查目录是否存在写权限，或换一个输出路径",
            ) from exc

    @classmethod
    def load(cls, path: str) -> "SplitResult":
        """从 JSON 文件读取切分结果。

        Args:
            path: ``save`` 生成的文件路径。

        Returns:
            还原后的 :class:`SplitResult`。

        Raises:
            SplitError: 文件不存在、JSON 损坏或结构不合法。
        """
        source = os.path.abspath(os.fspath(path))
        if not os.path.isfile(source):
            raise SplitError(
                f"切分结果文件不存在: {source}",
                hint="请先运行切分并 save()，或检查路径是否拼错",
            )
        try:
            with open(source, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise SplitError(
                f"切分结果读取失败: {source}（{exc}）",
                hint="检查文件权限",
            ) from exc
        except json.JSONDecodeError as exc:
            raise SplitError(
                f"切分结果不是合法 JSON: {source}（{exc}）",
                hint="请用 SplitResult.save() 重新生成，不要手工编辑",
            ) from exc
        if not isinstance(payload, Mapping):
            raise SplitError(
                f"切分结果顶层必须是映射: {source}（实际为 {type(payload).__name__}）",
                hint="期望结构: {splits, stats, mode, seed}",
            )
        splits_raw = payload.get("splits")
        if not isinstance(splits_raw, Mapping):
            raise SplitError(
                f"切分结果缺少合法的 splits 字段: {source}",
                hint="期望 {split: [下标, ...]}",
            )
        splits: dict[str, list[int]] = {}
        for name, indices in splits_raw.items():
            if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes, bytearray)):
                raise SplitError(
                    f"切分结果中 splits[{name!r}] 必须是下标列表",
                    hint="期望 {split: [下标, ...]}",
                )
            checked: list[int] = []
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise SplitError(
                        f"切分结果中 splits[{name!r}] 含非整数下标: {index!r}",
                        hint="下标必须是 Python 整数",
                    )
                if index < 0:
                    raise SplitError(
                        f"切分结果中 splits[{name!r}] 含负数下标: {index}",
                        hint="下标必须 >= 0",
                    )
                checked.append(index)
            splits[str(name)] = sorted(checked)
        stats = payload.get("stats")
        if not isinstance(stats, Mapping):
            raise SplitError(
                f"切分结果缺少合法的 stats 字段: {source}",
                hint="请用 SplitResult.save() 重新生成",
            )
        mode = payload.get("mode")
        if not isinstance(mode, str) or not mode:
            raise SplitError(
                f"切分结果缺少合法的 mode 字段: {source}",
                hint="mode 应为 ratio / dir / list",
            )
        seed = payload.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise SplitError(
                f"切分结果中的 seed 必须是整数或 null（当前为 {seed!r}）",
                hint="请用 SplitResult.save() 重新生成",
            )
        return cls(splits=splits, stats=dict(stats), mode=mode, seed=seed)

    def summary(self) -> str:
        """返回人类可读的多行摘要（模式、样本数、每类分布、warning）。"""
        header = f"切分模式: {self.mode}"
        if self.seed is not None:
            header += f"（seed={self.seed}）"
        lines = [header]
        stats = self.stats if isinstance(self.stats, Mapping) else {}
        total = stats.get("total")
        if not isinstance(total, int):
            total = sum(len(indices) for indices in self.splits.values())
        lines.append(f"总样本数: {total}")
        split_stats = stats.get("splits") if isinstance(stats.get("splits"), Mapping) else {}
        for name, indices in self.splits.items():
            info = split_stats.get(name) if isinstance(split_stats, Mapping) else None
            info = info if isinstance(info, Mapping) else {}
            count = info.get("count", len(indices))
            line = f"{name}: {count} 个样本"
            per_class = info.get("per_class")
            if isinstance(per_class, Mapping) and per_class:
                detail = "，".join(f"{key}={value}" for key, value in per_class.items())
                line += f"（{detail}）"
            lines.append(line)
        warnings = stats.get("warnings")
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) and warnings:
            lines.append(f"警告（{len(warnings)}）:")
            lines.extend(f"  - {warn}" for warn in warnings)
        return "\n".join(lines)


def build_splits(
    refs: "list[SampleRef]",
    config: "SplitConfig | Mapping[str, Any] | None",
    spec: "DVXDatasetSpec",
) -> SplitResult:
    """按配置把样本切成 train/val/test（契约 P1-5）。

    Args:
        refs: ``scan_samples`` 的输出；切分下标指向该列表的位置。
        config: 切分配置；为 None 时回退到 ``spec.splits``；
            也接受等价 mapping（``SplitConfig(**mapping)``）。
        spec: 数据集规格（用于类别表、多任务 task.path、root 与 spec 路径）。

    Returns:
        :class:`SplitResult`，其中 ``splits`` 的每个列表都升序且互不重叠、完整覆盖全部样本。

    Raises:
        SplitError: 配置缺失/非法、样本不足、路径未覆盖/重叠、切分不完整等。
    """
    if not refs:
        raise SplitError(
            "没有可切分的样本",
            hint="请检查 spec.root、parse.image_ext 与扫描逻辑；空数据集无法切分",
        )
    resolved = _resolve_config(config, spec)
    if resolved.mode == "ratio":
        return _build_ratio(refs, resolved, spec)
    if resolved.mode == "dir":
        return _build_dir(refs, resolved, spec)
    if resolved.mode == "list":
        return _build_list(refs, resolved, spec)
    raise SplitError(
        f"非法的切分模式: {resolved.mode!r}",
        hint=f"可选值: {list(SPLIT_MODES)}",
    )


# ---------------------------------------------------------------------------
# 三种模式
# ---------------------------------------------------------------------------


def _build_ratio(
    refs: "list[SampleRef]", config: SplitConfig, spec: "DVXDatasetSpec"
) -> SplitResult:
    """ratio 模式：按 rel_path 稳定排序 + seed 打乱 + 可选分层。"""
    n = len(refs)
    ratio = _checked_ratio(config.ratio)
    names = _canonical_names(ratio)
    if n < len(names):
        raise SplitError(
            f"样本数 {n} 少于切分数 {len(names)}",
            hint="请减少切分数量（例如只用 train/val）或补充样本",
        )
    if config.stratify not in (None, "class", "none"):
        raise SplitError(
            f"非法的 splits.stratify: {config.stratify!r}",
            hint="可选值: class / none",
        )
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise SplitError(
            f"splits.seed 必须是整数（当前为 {config.seed!r}）",
            hint="seed 决定切分可复现性，例如 seed: 42",
        )
    mode_seed: int | None = config.seed

    # 稳定基准：按 rel_path 排序，rng 只作用于排序后的位置，输入顺序不影响结果
    order = sorted(range(n), key=lambda index: _rel_path(refs[index]))
    rng = random.Random(config.seed)
    warnings: list[str] = []
    assignment: dict[int, str] = {}

    if config.stratify == "class":
        groups: dict[Any, list[int]] = {}
        for index in order:
            groups.setdefault(_stratify_key(refs[index]), []).append(index)
        group_keys: list[Any] = sorted(groups, key=_group_sort_key)
    else:
        groups = {None: list(order)}
        group_keys = [None]

    for key in group_keys:
        indices = groups[key]
        rng.shuffle(indices)
        quotas = _allocate_quotas(len(indices), names, ratio)
        cursor = 0
        for name, quota in zip(names, quotas):
            for sample_index in indices[cursor : cursor + quota]:
                assignment[sample_index] = name
            cursor += quota
        if cursor < len(indices):
            # 理论上 ratio 之和已校验为 1；极端浮点情况下把余数并入最后一个 split
            for sample_index in indices[cursor:]:
                assignment[sample_index] = names[-1]
            warnings.append(
                f"ratio 浮点余数导致 {len(indices) - cursor} 个样本按兜底规则并入 "
                f"{names[-1]}（类别组 {key!r}）"
            )

    splits = {name: [] for name in names}
    for sample_index, name in assignment.items():
        splits[name].append(sample_index)

    if not splits["train"]:
        donors = [name for name in names if name != "train" and splits[name]]
        if not donors:
            raise SplitError(
                "无法保证 train 非空",
                hint="请检查 ratio 配置是否给 train 留了正比例",
            )
        donor = max(donors, key=lambda name: (len(splits[name]), -names.index(name)))
        moved = splits[donor].pop()
        splits["train"].append(moved)
        warnings.append(
            f"train 切分为空，已从 {donor} 移动 1 个样本到 train"
            f"（rel_path={_rel_path(refs[moved])}）；建议提高 train 比例或补充样本"
        )

    return _finalize(
        refs,
        splits,
        names,
        mode="ratio",
        seed=mode_seed,
        warnings=warnings,
        spec=spec,
    )


def _build_dir(refs: "list[SampleRef]", config: SplitConfig, spec: "DVXDatasetSpec") -> SplitResult:
    """dir 模式：按目录映射判定样本归属。"""
    dir_map = config.dir_map
    if not isinstance(dir_map, Mapping) or not dir_map:
        raise SplitError(
            "mode=dir 时必须提供非空的 splits.dir_map",
            hint="例如 dir_map: {train: train_data/train, val: train_data/val}",
        )
    unknown = [name for name in dir_map if name not in SPLIT_NAMES]
    if unknown:
        raise SplitError(
            f"splits.dir_map 含非法切分名: {unknown}",
            hint=f"只允许 {list(SPLIT_NAMES)}",
        )
    names = _canonical_names(dir_map)
    _ensure_train(names, "splits.dir_map")
    normalized: dict[str, str] = {}
    for name, subdir in dir_map.items():
        if not isinstance(subdir, str) or not subdir.strip():
            raise SplitError(
                f"splits.dir_map[{name!r}] 必须是相对目录字符串（当前为 {subdir!r}）",
                hint="例如 train: train_data/train",
            )
        if posixpath.isabs(subdir.replace("\\", "/")):
            raise SplitError(
                f"splits.dir_map[{name!r}] 必须是相对目录: {subdir!r}",
                hint="dir_map 相对 spec.root（多任务时相对 task.path），不能写绝对路径",
            )
        normalized[name] = _normalize_rel(subdir)

    task_paths = {
        task.name: _normalize_rel(task.path)
        for task in (getattr(spec, "tasks", None) or [])
        if isinstance(getattr(task, "name", None), str)
    }
    is_multi_task = getattr(spec, "data_type", "") == "multi_task_classification"
    splits = {name: [] for name in names}
    unmatched: list["SampleRef"] = []
    warnings: list[str] = []

    for sample_index, ref in enumerate(refs):
        rel = _rel_path(ref)
        target = rel
        if is_multi_task:
            task_name = getattr(ref, "task", None)
            task_prefix = task_paths.get(task_name) if isinstance(task_name, str) else None
            if task_prefix is None:
                unmatched.append(ref)
                continue
            stripped = _strip_prefix(rel, task_prefix)
            if stripped is None:
                unmatched.append(ref)
                continue
            target = stripped
        matches = [name for name in names if _is_under(target, normalized[name])]
        if not matches:
            unmatched.append(ref)
            continue
        if len(matches) > 1:
            matches.sort(key=lambda name: len(normalized[name]), reverse=True)
            warnings.append(
                f"样本 {rel} 同时命中多个 split 目录 {matches}，按最长目录归属 {matches[0]}"
            )
        splits[matches[0]].append(sample_index)

    if unmatched:
        samples = [_rel_path(ref) for ref in unmatched[:5]]
        raise SplitError(
            f"{len(unmatched)} 个样本未被 dir_map 覆盖",
            hint=f"前 5 个样例: {samples}；dir_map={dict(dir_map)}"
            "（多任务时目录相对每个 task.path）",
        )
    return _finalize(
        refs,
        splits,
        names,
        mode="dir",
        seed=None,
        warnings=warnings,
        spec=spec,
    )


def _build_list(
    refs: "list[SampleRef]", config: SplitConfig, spec: "DVXDatasetSpec"
) -> SplitResult:
    """list 模式：读取显式路径清单并做严格归属校验。"""
    list_file = _resolve_list_file(config.list_file, spec)
    payload = _load_list_file(list_file)
    names_raw = list(payload)
    unknown = [name for name in names_raw if name not in SPLIT_NAMES]
    if unknown:
        raise SplitError(
            f"清单文件含非法切分名: {unknown}",
            hint=f"只允许 {list(SPLIT_NAMES)}；出错文件: {list_file}",
        )
    names = _canonical_names(payload)
    _ensure_train(names, "清单文件")

    by_rel: dict[str, int] = {}
    for sample_index, ref in enumerate(refs):
        rel = _rel_path(ref)
        if rel in by_rel:
            raise SplitError(
                f"refs 中存在重复 rel_path: {rel}",
                hint="同一相对路径只能对应一个样本；请检查扫描逻辑",
            )
        by_rel[rel] = sample_index

    splits = {name: [] for name in names}
    listed: dict[int, str] = {}
    missing_paths: list[str] = []
    for name in names:
        entries = payload[name]
        if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence):
            raise SplitError(
                f"清单文件 splits[{name!r}] 必须是路径列表",
                hint=f"期望 {{train: [相对路径, ...]}}；出错文件: {list_file}",
            )
        seen_in_split: set[str] = set()
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise SplitError(
                    f"清单文件 splits[{name!r}] 含非字符串路径: {entry!r}",
                    hint=f"出错文件: {list_file}",
                )
            rel = _normalize_rel(entry)
            if rel in seen_in_split:
                raise SplitError(
                    f"路径在 splits[{name!r}] 中重复: {rel}",
                    hint=f"每个样本只能出现一次；出错文件: {list_file}",
                )
            seen_in_split.add(rel)
            sample_index = by_rel.get(rel)
            if sample_index is None:
                missing_paths.append(f"{name}: {rel}")
                continue
            if sample_index in listed:
                raise SplitError(
                    f"样本同时出现在 {listed[sample_index]} 与 {name}: {rel}",
                    hint="清单文件中的切分必须互斥；请删除重复条目",
                )
            listed[sample_index] = name
            splits[name].append(sample_index)

    if missing_paths:
        raise SplitError(
            f"清单文件中有 {len(missing_paths)} 个路径在数据集中不存在",
            hint=f"前 5 个: {missing_paths[:5]}；清单文件: {list_file}"
            "（路径相对 spec.root，posix 风格）",
        )
    unlisted = [index for index in range(len(refs)) if index not in listed]
    if unlisted:
        samples = [_rel_path(refs[index]) for index in unlisted[:5]]
        raise SplitError(
            f"有 {len(unlisted)} 个样本未出现在清单文件中",
            hint=f"前 5 个: {samples}；清单文件: {list_file}",
        )
    return _finalize(
        refs,
        splits,
        names,
        mode="list",
        seed=None,
        warnings=[],
        spec=spec,
    )


# ---------------------------------------------------------------------------
# 配置解析与路径小工具
# ---------------------------------------------------------------------------


def _resolve_config(
    config: "SplitConfig | Mapping[str, Any] | None", spec: "DVXDatasetSpec"
) -> SplitConfig:
    """把 None / mapping 归一化为 SplitConfig。"""
    resolved: Any = config if config is not None else getattr(spec, "splits", None)
    if resolved is None:
        raise SplitError(
            "缺少切分配置",
            hint="请在 dvx_dataset.yaml 中配置 splits，或向 build_splits 传入 SplitConfig",
        )
    if isinstance(resolved, Mapping):
        try:
            resolved = SplitConfig(**dict(resolved))
        except TypeError as exc:
            raise SplitError(
                f"非法的切分配置: {exc}",
                hint="可用字段: mode / ratio / seed / stratify / list_file / dir_map",
            ) from exc
    if not isinstance(resolved, SplitConfig):
        raise SplitError(
            f"config 必须是 SplitConfig / mapping / None（当前为 {type(resolved).__name__}）",
            hint="通常直接传 spec.splits",
        )
    return resolved


def _checked_ratio(ratio: Any) -> dict[str, float]:
    """校验并返回 float 化的 ratio。"""
    if not isinstance(ratio, Mapping) or not ratio:
        raise SplitError(
            "splits.ratio 必须是非空映射",
            hint="例如 {train: 0.8, val: 0.2}",
        )
    unknown = [name for name in ratio if name not in SPLIT_NAMES]
    if unknown:
        raise SplitError(
            f"splits.ratio 含非法切分名: {unknown}",
            hint=f"只允许 {list(SPLIT_NAMES)}",
        )
    if "train" not in ratio:
        raise SplitError(
            "splits.ratio 缺少 train",
            hint="切分必须包含 train（例如 train: 0.8, val: 0.2）",
        )
    result: dict[str, float] = {}
    total = 0.0
    for name, value in ratio.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitError(
                f"splits.ratio[{name!r}] 必须是数字（当前为 {value!r}）",
                hint="例如 train: 0.8",
            )
        if value < 0:
            raise SplitError(
                f"splits.ratio[{name!r}] 不能为负数（当前为 {value!r}）",
                hint="比例应为 0~1 之间的数",
            )
        result[name] = float(value)
        total += float(value)
    if abs(total - 1.0) > RATIO_SUM_TOLERANCE:
        raise SplitError(
            f"splits.ratio 之和必须为 1.0（当前为 {total:.9f}）",
            hint=f"容差 {RATIO_SUM_TOLERANCE}，例如 train: 0.8, val: 0.2",
        )
    return result


def _canonical_names(names: Any) -> list[str]:
    """按 {train, val, test} 的固定顺序返回切分名列表。"""
    available = list(names)
    return [name for name in SPLIT_NAMES if name in available]


def _ensure_train(names: Sequence[str], where: str) -> None:
    """要求切分名里包含 train。"""
    if "train" not in names:
        raise SplitError(
            f"{where} 缺少 train 切分",
            hint=f"必须包含 train；当前切分名: {list(names)}",
        )


def _allocate_quotas(size: int, names: Sequence[str], ratio: Mapping[str, float]) -> list[int]:
    """用最大余数法把 size 个样本按 ratio 分配到各 split（结果之和恒为 size）。"""
    quotas = [ratio[name] * size for name in names]
    base = [int(math.floor(quota + 1e-12)) for quota in quotas]
    remainder = size - sum(base)
    fractions = [quota - base[index] for index, quota in enumerate(quotas)]
    ranking = sorted(range(len(names)), key=lambda index: (-fractions[index], index))
    cursor = 0
    while remainder > 0 and ranking:
        base[ranking[cursor % len(ranking)]] += 1
        remainder -= 1
        cursor += 1
    return base


def _stratify_key(ref: "SampleRef") -> Any:
    """分层键：label；多任务为 (task, label)；无 label 的样本单独成组。"""
    label = _label_of(ref)
    task = getattr(ref, "task", None)
    if label is None:
        return "__unlabeled__" if task is None else (task, "__unlabeled__")
    if task is None:
        return label
    return (task, label)


def _group_sort_key(key: Any) -> str:
    """让混合类型的组键也能稳定排序。"""
    return str(key)


def _label_of(ref: "SampleRef") -> int | None:
    """读取样本整数标签（兼容 numpy 整数），无标签返回 None。"""
    raw = getattr(ref, "label", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _rel_path(ref: "SampleRef") -> str:
    """返回样本的 posix 风格相对路径（内部统一用 /）。"""
    value = getattr(ref, "rel_path", None)
    if not isinstance(value, str) or not value:
        raise SplitError(
            f"样本缺少 rel_path: {ref!r}",
            hint="SampleRef.rel_path 用于切分与清单匹配，必须是非空字符串",
        )
    return _normalize_rel(value)


def _normalize_rel(value: str) -> str:
    """把路径规范成 posix 相对路径（去掉 ./、重复斜杠、尾斜杠）。"""
    text = value.replace("\\", "/").strip()
    text = posixpath.normpath(text)
    if text == ".":
        return ""
    return text.lstrip("/")


def _strip_prefix(rel: str, prefix: str) -> str | None:
    """若 rel 位于 prefix 目录下，返回去掉前缀后的相对路径；否则 None。"""
    if not prefix:
        return rel
    if rel == prefix:
        return ""
    if rel.startswith(prefix + "/"):
        return rel[len(prefix) + 1 :]
    return None


def _is_under(rel: str, prefix: str) -> bool:
    """判断 rel 是否位于 prefix 目录下（按路径分量匹配，避免前缀误伤）。"""
    if not prefix:
        return True
    return rel == prefix or rel.startswith(prefix + "/")


def _resolve_list_file(list_file: Any, spec: "DVXDatasetSpec") -> str:
    """解析 list_file 绝对路径（相对 spec 文件所在目录）。"""
    if not isinstance(list_file, str) or not list_file.strip():
        raise SplitError(
            "mode=list 时必须提供 splits.list_file",
            hint="list_file 是 JSON/YAML 清单路径（相对 spec 文件所在目录或绝对路径）",
        )
    value = list_file.strip()
    if os.path.isabs(value):
        return os.path.normpath(value)
    spec_path = getattr(spec, "spec_path", "") or ""
    base = os.path.dirname(os.path.abspath(spec_path)) if spec_path else os.getcwd()
    return os.path.normpath(os.path.abspath(os.path.join(base, value)))


def _load_list_file(path: str) -> dict[str, Any]:
    """读取并解析 JSON/YAML 清单文件。"""
    if not os.path.isfile(path):
        raise SplitError(
            f"切分清单文件不存在: {path}",
            hint="请确认 splits.list_file 路径（相对 spec 文件所在目录）",
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError as exc:
        raise SplitError(
            f"切分清单文件读取失败: {path}（{exc}）",
            hint="请检查文件权限与编码（UTF-8）",
        ) from exc
    except yaml.YAMLError as exc:
        raise SplitError(
            f"切分清单文件解析失败: {path}（{exc}）",
            hint="JSON 或 YAML 均可，但结构必须是 {split: [相对路径, ...]}",
        ) from exc
    if not isinstance(payload, Mapping):
        raise SplitError(
            f"切分清单顶层必须是映射: {path}（实际为 {type(payload).__name__}）",
            hint="期望结构: {train: [rel/path], val: [rel/path]}",
        )
    if not payload:
        raise SplitError(
            f"切分清单为空: {path}",
            hint="至少需要 train 切分",
        )
    return dict(payload)


# ---------------------------------------------------------------------------
# 结果汇总与硬校验
# ---------------------------------------------------------------------------


def _finalize(
    refs: "list[SampleRef]",
    splits: dict[str, list[int]],
    names: Sequence[str],
    *,
    mode: str,
    seed: int | None,
    warnings: list[str],
    spec: "DVXDatasetSpec",
) -> SplitResult:
    """统一做重叠/覆盖校验，排序下标，生成 stats 与 SplitResult。"""
    total = len(refs)
    seen: set[int] = set()
    for name in names:
        indices = splits[name]
        duplicates = len(indices) - len(set(indices))
        if duplicates:
            raise SplitError(
                f"切分 {name} 内部出现 {duplicates} 个重复下标",
                hint="这是切分实现缺陷，请报告；也可检查 refs 是否有重复项",
            )
        overlap = seen.intersection(indices)
        if overlap:
            raise SplitError(
                f"切分 {name} 与其它切分重叠 {len(overlap)} 个样本",
                hint=f"重叠下标样例: {sorted(overlap)[:5]}；请检查 dir_map / 清单文件",
            )
        seen.update(indices)
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise SplitError(
                    f"切分 {name} 含非法下标: {index!r}",
                    hint="下标必须是 >= 0 的整数",
                )
            if index >= total:
                raise SplitError(
                    f"切分 {name} 含越界下标: {index}（样本数 {total}）",
                    hint="下标必须落在 [0, len(refs)) 内",
                )
        splits[name] = sorted(indices)

    missing = sorted(set(range(total)) - seen)
    if missing:
        samples = [_rel_path(refs[index]) for index in missing[:5]]
        raise SplitError(
            f"有 {len(missing)} 个样本未被任何切分覆盖",
            hint=f"缺失样例: {samples}；请检查 dir_map / 清单文件是否覆盖全部样本",
        )

    stats_warnings = list(warnings)
    empty_warnings = [f"切分 {name} 为空（0 个样本）" for name in names if not splits[name]]
    stats_warnings.extend(empty_warnings)

    per_class_by_split = {name: _per_class_counts(refs, splits[name], spec) for name in names}
    stats_warnings.extend(_class_warnings(per_class_by_split, names, spec))

    stats = {
        "total": total,
        "splits": {
            name: {"count": len(splits[name]), "per_class": per_class_by_split[name]}
            for name in names
        },
        "warnings": stats_warnings,
    }
    return SplitResult(splits=splits, stats=stats, mode=mode, seed=seed)


def _per_class_counts(
    refs: "list[SampleRef]", indices: Sequence[int], spec: "DVXDatasetSpec"
) -> dict[str, int]:
    """统计一个 split 内每个类别的样本数（仅分类类数据集有值）。"""
    data_type = getattr(spec, "data_type", "")
    if data_type not in ("classification", "multi_task_classification"):
        return {}
    class_names = list(getattr(spec, "classes", None) or [])
    task_classes = {
        task.name: list(task.classes)
        for task in (getattr(spec, "tasks", None) or [])
        if isinstance(getattr(task, "name", None), str)
    }
    counts: dict[str, int] = {}
    for index in indices:
        key = _class_key(refs[index], data_type, class_names, task_classes)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _class_key(
    ref: "SampleRef",
    data_type: str,
    class_names: Sequence[str],
    task_classes: Mapping[str, Sequence[str]],
) -> str | None:
    """把样本映射为 per_class 统计使用的类别键（多任务为 ``task:class``）。"""
    label = _label_of(ref)
    if label is None:
        return None
    if data_type == "multi_task_classification":
        task = getattr(ref, "task", None)
        if not isinstance(task, str) or not task:
            return f"<unknown-task>:<label:{label}>"
        names = task_classes.get(task, [])
        name = names[label] if 0 <= label < len(names) else f"<label:{label}>"
        return f"{task}:{name}"
    name = class_names[label] if 0 <= label < len(class_names) else f"<label:{label}>"
    return name


def _class_warnings(
    per_class_by_split: Mapping[str, Mapping[str, int]],
    names: Sequence[str],
    spec: "DVXDatasetSpec",
) -> list[str]:
    """对"整个数据集缺失"与"val/test 中缺失"的类别生成 warning。"""
    universe = _class_universe(spec)
    if not universe:
        return []
    totals: dict[str, int] = {key: 0 for key in universe}
    for counts in per_class_by_split.values():
        for key, count in counts.items():
            totals[key] = totals.get(key, 0) + count
    warnings: list[str] = []
    for key in universe:
        if totals.get(key, 0) == 0:
            warnings.append(f"类别 {key} 在整个数据集中没有样本")
            continue
        for name in names:
            if name == "train":
                continue
            if per_class_by_split.get(name, {}).get(key, 0) == 0:
                warnings.append(f"类别 {key} 在 {name} 中没有样本")
    return warnings


def _class_universe(spec: "DVXDatasetSpec") -> list[str]:
    """返回 per_class 统计使用的全部类别键（仅分类类数据集）。"""
    data_type = getattr(spec, "data_type", "")
    if data_type == "multi_task_classification":
        return [
            f"{task.name}:{name}"
            for task in (getattr(spec, "tasks", None) or [])
            for name in task.classes
        ]
    if data_type == "classification":
        return list(getattr(spec, "classes", None) or [])
    return []
