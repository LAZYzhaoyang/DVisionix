# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 四个真实数据集的验收测试（契约 P1-11；只依赖 scan_samples + build_splits）。
"""真实数据集验收测试（契约 P1-10 / P1-11）。

只使用 ``load_spec`` + ``scan_samples`` + ``build_splits`` 验证四套真实数据，
**不依赖** ``DVXDataset``（由并行实现提供），也**不修改** ``project/data`` 下任何文件。

数据位置按以下顺序解析，任一 spec 不存在时对应用例 ``skipif``：

1. 环境变量 ``DVX_DATA_ROOT``；
2. ``<repo>/project/data``；
3. ``<repo>/../project/data``（本仓库的实际布局：数据与代码仓库同级）。

覆盖的硬指标：

- cls：2210 张，5 类（顺序固定），80/20 分层切分 1768/442；
- det：1646 张、20 类（排序）、10014 个框，80/20 切分 1317/329；
- multicls：2458 张（licence 1076 / logo 1382），ratio 切分 1967/491；
- seg：766 张、18 类（__background__ + 17 类）、4348 个多边形，80/20 切分 613/153。
"""

from __future__ import annotations

import functools
import os
from collections import Counter
from pathlib import Path

import pytest

from dvisionix.data.spec.scan import SampleRef, scan_samples
from dvisionix.data.spec.schema import DVXDatasetSpec, load_spec
from dvisionix.data.spec.split import SplitResult, build_splits

pytestmark = pytest.mark.integration

#: 仓库根目录（tests/test_data/test_acceptance_real_data.py 的上两级）
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 四个真实数据集目录名（同时也是 spec 所在子目录名）
DATASET_NAMES = ("cls", "det", "multicls", "seg")

#: 各数据集期望的样本总数
EXPECTED_TOTAL = {"cls": 2210, "det": 1646, "multicls": 2458, "seg": 766}

#: 各数据集在 spec 默认切分下的样本数
EXPECTED_SPLITS = {
    "cls": {"train": 1768, "val": 442},
    "det": {"train": 1317, "val": 329},
    "multicls": {"train": 1967, "val": 491},
    "seg": {"train": 613, "val": 153},
}

#: cls 的固定类别顺序（顺序即 label id）
CLS_CLASSES = ["幻影灰", "新卡其白", "新碳晶黑", "科技灰", "纳斯达克银"]

#: det 的 20 类（全部 rectangle label 排序后的并集）
DET_CLASSES = [
    "7",
    "7_large",
    "C",
    "Che",
    "G",
    "H1",
    "H2",
    "H3",
    "I",
    "MAX",
    "O",
    "O_small",
    "P",
    "Qi1",
    "Qi3",
    "R",
    "Rui",
    "S",
    "T",
    "V_RED",
]

#: seg 的 17 个 polygon label（排序后），类表还需在最前面加 __background__
SEG_LABELS = [
    "ABar",
    "airBag",
    "bumper",
    "fangXiangPan",
    "grille",
    "licence",
    "light",
    "logo",
    "midLight",
    "rearWindow",
    "riper",
    "sideLogo",
    "sideReflector",
    "spoiler",
    "stopLight",
    "upSideBar",
    "weatherstrip",
]


def _resolve_data_root() -> Path:
    """定位 ``project/data``：环境变量优先，其次仓库内、再次仓库同级。"""
    candidates: list[Path] = []
    override = os.environ.get("DVX_DATA_ROOT")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            _REPO_ROOT / "project" / "data",
            _REPO_ROOT.parent / "project" / "data",
        ]
    )
    for candidate in candidates:
        if (candidate / "cls" / "dvx_dataset.yaml").is_file():
            return candidate
    # 都找不到时返回默认位置，让各用例按真实路径 skip
    return candidates[-1]


#: 真实数据根目录（不存在时用例 skip）
DATA_ROOT = _resolve_data_root()


def _spec_path(name: str) -> Path:
    """返回某个数据集的 spec 路径。"""
    return DATA_ROOT / name / "dvx_dataset.yaml"


def _skip_missing(name: str):
    """返回"spec 缺失则跳过"的 pytest mark。"""
    return pytest.mark.skipif(
        not _spec_path(name).is_file(),
        reason=f"真实数据 spec 不存在，跳过验收: {_spec_path(name)}",
    )


@functools.lru_cache(maxsize=None)
def _load(name: str) -> DVXDatasetSpec:
    """加载并缓存 spec（四个数据集在进程内只解析一次）。"""
    return load_spec(str(_spec_path(name)))


@functools.lru_cache(maxsize=None)
def _scan(name: str) -> list[SampleRef]:
    """扫描并缓存样本列表（真实数据只读，缓存不会改变结果）。"""
    return scan_samples(_load(name))


def _assert_partition(result: SplitResult, total: int) -> None:
    """校验切分结果：互不重叠、完整覆盖、下标升序且 train 非空。"""
    assert set(result.splits) == {"train", "val"}
    assert result.splits["train"]
    covered: list[int] = []
    for indices in result.splits.values():
        assert indices == sorted(indices), "切分下标必须升序"
        covered.extend(indices)
    assert len(covered) == len(set(covered)), "切分之间不得重叠"
    assert set(covered) == set(range(total)), "切分必须完整覆盖全部样本"
    assert result.stats["total"] == total


# ---------------------------------------------------------------------------
# 扫描：样本数与类别表
# ---------------------------------------------------------------------------


@_skip_missing("cls")
def test_cls_scan() -> None:
    """cls：2210 张 5 类，类别顺序固定，每类数量与磁盘一致。"""
    spec = _load("cls")
    assert spec.data_type == "classification"
    assert spec.layout == "folder_per_class"
    assert spec.class_names() == CLS_CLASSES

    refs = _scan("cls")
    assert len(refs) == EXPECTED_TOTAL["cls"]
    assert [ref.rel_path for ref in refs] == sorted(ref.rel_path for ref in refs)
    assert len({ref.rel_path for ref in refs}) == len(refs)
    assert all(ref.label is not None for ref in refs)

    counts = Counter(ref.label for ref in refs)
    assert [counts[index] for index in range(len(CLS_CLASSES))] == [530, 601, 609, 193, 277]


@_skip_missing("det")
def test_det_scan() -> None:
    """det：1646 张、20 类、10014 个框，每张图片都有同名 labelme JSON。"""
    spec = _load("det")
    assert spec.data_type == "detection"
    assert spec.layout == "labelme"
    assert spec.class_names() == DET_CLASSES

    refs = _scan("det")
    assert len(refs) == EXPECTED_TOTAL["det"]
    assert all(ref.ann_path and Path(ref.ann_path).is_file() for ref in refs)
    assert all(ref.image_size is not None for ref in refs)
    assert all(ref.boxes is not None and ref.labels is not None for ref in refs)
    assert all(ref.boxes.shape[1] == 4 for ref in refs)
    assert all(len(ref.boxes) == len(ref.labels) for ref in refs)
    assert sum(len(ref.boxes) for ref in refs) == 10014
    assert sum(ref.meta.get("skipped_shapes", 0) for ref in refs) == 0


@_skip_missing("multicls")
def test_multicls_scan() -> None:
    """multicls：双任务共 2458 张（licence 1076 / logo 1382），任务与类表正确。"""
    spec = _load("multicls")
    assert spec.data_type == "multi_task_classification"
    assert spec.layout == "folder_per_class"
    assert [task.name for task in spec.tasks] == ["licence", "logo"]
    assert [task.path for task in spec.tasks] == [
        "T15C_1CCD6_licence",
        "T15C_1CCD6_logo",
    ]
    assert spec.class_names("licence") == ["无", "短牌照", "长牌照"]
    assert spec.class_names("logo") == ["一体奇瑞标", "嵌入奇瑞标"]
    # 标准多任务结构 <任务>/<类别>/*.jpg，不使用 dir_map / 额外 subdir
    assert spec.splits is not None and spec.splits.mode == "ratio"
    assert spec.splits.dir_map is None

    refs = _scan("multicls")
    assert len(refs) == EXPECTED_TOTAL["multicls"]
    assert len({ref.rel_path for ref in refs}) == len(refs)
    task_paths = {task.name: task.path for task in spec.tasks}
    assert all(ref.rel_path.startswith(f"{task_paths[ref.task]}/") for ref in refs)
    assert not any("train_data/" in ref.rel_path for ref in refs)

    licence = Counter(ref.label for ref in refs if ref.task == "licence")
    logo = Counter(ref.label for ref in refs if ref.task == "logo")
    assert sum(licence.values()) == 1076
    assert sum(logo.values()) == 1382
    assert [licence[index] for index in range(3)] == [442, 30, 604]
    assert [logo[index] for index in range(2)] == [856, 526]


@_skip_missing("seg")
def test_seg_scan() -> None:
    """seg：766 张、18 类（__background__ + 17 类）、4348 个多边形。"""
    spec = _load("seg")
    assert spec.data_type == "segmentation"
    assert spec.layout == "labelme"
    assert spec.seg_mode == "semantic"
    assert spec.ignore_index == 255
    assert spec.class_names() == ["__background__"] + SEG_LABELS

    refs = _scan("seg")
    assert len(refs) == EXPECTED_TOTAL["seg"]
    assert all(ref.ann_path and Path(ref.ann_path).is_file() for ref in refs)
    assert all(ref.image_size is not None for ref in refs)
    assert all(ref.segments is not None and ref.segment_labels is not None for ref in refs)
    assert all(len(ref.segments) == len(ref.segment_labels) for ref in refs)
    assert sum(len(ref.segment_labels) for ref in refs) == 4348
    assert sum(ref.meta.get("skipped_shapes", 0) for ref in refs) == 0
    # 背景不来自标注形状：所有 segment label 都应落在 1..17
    assert all(int(label) > 0 for ref in refs for label in ref.segment_labels)


# ---------------------------------------------------------------------------
# 切分：默认 splits 的样本数与自洽性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=_skip_missing(name), id=name) for name in DATASET_NAMES],
)
def test_real_data_splits(name: str) -> None:
    """四套真实数据的默认切分：样本数固定、互不重叠且完整覆盖。"""
    spec = _load(name)
    refs = _scan(name)
    assert spec.splits is not None

    result = build_splits(refs, spec.splits, spec)
    assert {split: len(indices) for split, indices in result.splits.items()} == EXPECTED_SPLITS[
        name
    ]
    assert result.mode == spec.splits.mode
    if spec.splits.mode == "ratio":
        assert result.seed == spec.splits.seed
    else:
        assert result.seed is None
    _assert_partition(result, len(refs))

    # 确定性：同一输入重复构建必须完全一致
    again = build_splits(refs, spec.splits, spec)
    assert again.splits == result.splits
