# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 官方配置端到端门禁：configs/ 下每个可训练配置都必须能跑完 1 个 epoch。
"""官方配置端到端门禁（v1.1 阶段 0 新增）。

为什么需要它（见 CodePlan 7.1.2 的 D20）：
v1.0.0 的 286 条测试只覆盖「组件」与「build_model()」，**从不构建 loss、从不执行
validation_step、从不跑训练循环**。于是出现了一个危险组合：

- 单元/组件测试全绿；
- 官方配置 `simclr_synthetic` / `centernet_synthetic` / `yolov10_synthetic` 实际无法训练；
- 测试文件 `test_v010_direction3.py` 甚至点名了 centernet 与 yolov10，却只断言
  ``build_model(...) is not None``。

本文件补上「装配层」这道网：对 ``configs/`` 下每个可训练配置，完整执行
``tools/train.py`` 的 ``main()``（真实解析 argv、真实构建 data/model/task/trainer、
真实训练 1 个 epoch），并断言产物 ``history.csv`` 存在。

使用：
    pytest tests/test_e2e_configs.py -q            # 全部配置
    pytest tests -q -m "not slow"                  # 跳过本门禁（快速反馈）
    pytest tests/test_e2e_configs.py -q -k yolov10 # 单个配置
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIGS_ROOT = os.path.join(ROOT, "configs")

#: 前置条件型模板配置：不参与 E2E 门禁。
#: - hparam_search 由 tools/hparam_search.py 消费，缺少 task_type 属预期
#: - linear_eval 需要手工提供预训练 checkpoint 路径（当前为占位符）
TEMPLATE_CONFIGS = frozenset(
    {
        "classification/hparam_search.yaml",
        "classification/linear_eval.yaml",
    }
)

#: 已知无法训练的配置 —— 机制保留，供未来新配置的临时豁免使用。
#: 登记时必须带 ``xfail(strict=True)``，因此一旦修复就会立刻变成 XPASS 失败，
#: 强制把条目移除，不会长期遗留。
#:
#: v1.0.0 基线曾有的三条（simclr / centernet / yolov10）已于 CodePlan 7.3 阶段 1 全部修复。
KNOWN_BROKEN = frozenset()


def _discover_configs() -> list:
    """递归发现 configs/ 下所有参与门禁的 YAML 配置（POSIX 风格相对路径，已排序）。"""
    found = []
    for dirpath, _dirnames, filenames in os.walk(CONFIGS_ROOT):
        for name in sorted(filenames):
            if not name.endswith(".yaml"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), CONFIGS_ROOT)
            rel = rel.replace(os.sep, "/")
            if rel in TEMPLATE_CONFIGS:
                continue
            found.append(rel)
    return sorted(found)


TRAINABLE_CONFIGS = _discover_configs()


def _param(rel: str):
    marks = ()
    if rel in KNOWN_BROKEN:
        marks = (
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{rel} 在 v1.0.0 无法训练，属阶段 1 待修清单（CodePlan 7.3）；"
                    f"修复后必须从 KNOWN_BROKEN 移除"
                ),
            ),
        )
    return pytest.param(rel, marks=marks, id=rel.replace("/", "__").replace(".yaml", ""))


@pytest.mark.integration
def test_config_discovery_is_sane():
    """自检：配置发现逻辑不能悄悄退化成空集合或漏掉目录。"""
    assert len(TRAINABLE_CONFIGS) >= 15, (
        f"只发现 {len(TRAINABLE_CONFIGS)} 个可训练配置，疑似发现逻辑被破坏：" f"{TRAINABLE_CONFIGS}"
    )
    for rel in TRAINABLE_CONFIGS:
        assert os.path.isfile(os.path.join(CONFIGS_ROOT, rel)), rel
    # 三个任务域都必须被覆盖，否则门禁会漏掉整个任务类型
    for task_dir in ("classification/", "detection/", "segmentation/"):
        assert any(rel.startswith(task_dir) for rel in TRAINABLE_CONFIGS), task_dir


@pytest.mark.unit
@pytest.mark.parametrize("task_type", ["classification", "detection", "segmentation", "simclr"])
def test_synthetic_train_and_val_splits_do_not_overlap(task_type, tmp_path):
    """D13：合成数据的 train / val 不能共用样本文件。

    v1.0.0 两者共用同一目录与 ``img_%04d.png`` 命名，于是 val 的前 N 张
    恰好就是 train 的前 N 张 —— 所有 demo 的 ``val_loss`` / ``val_acc``
    都带数据泄漏，指标偏乐观且无意义。
    """
    from tools import train as train_tool

    cache = str(tmp_path / "cache")
    train_ds = train_tool.build_synthetic_dataset(task_type, 4, 3, 32, None, cache, split="train")
    val_ds = train_tool.build_synthetic_dataset(task_type, 4, 3, 32, None, cache, split="val")

    def _paths(ds):
        keys = ["image"] + (["mask"] if task_type == "segmentation" else [])
        return {tuple(os.path.basename(s[k]) for k in keys) for s in ds.samples}

    train_paths, val_paths = _paths(train_ds), _paths(val_ds)
    assert len(train_paths) == 4 and len(val_paths) == 4
    assert not (train_paths & val_paths), f"train/val 共用样本：{train_paths & val_paths}"


@pytest.mark.unit
def test_synthetic_dataset_is_reproducible(tmp_path):
    """同一 split 重复构建必须得到完全相同的样本路径与标签（便于复现与缓存）。"""
    from tools import train as train_tool

    cache = str(tmp_path / "cache")
    first = train_tool.build_synthetic_dataset("classification", 3, 3, 32, None, cache, "train")
    second = train_tool.build_synthetic_dataset("classification", 3, 3, 32, None, cache, "train")
    assert [s["image"] for s in first.samples] == [s["image"] for s in second.samples]
    assert [s["label"] for s in first.samples] == [s["label"] for s in second.samples]


@pytest.mark.unit
def test_synthetic_dataset_rejects_unknown_task_type(tmp_path):
    from tools import train as train_tool

    with pytest.raises(ValueError, match="不支持 task_type"):
        train_tool.build_synthetic_dataset("nope", 2, 3, 32, None, str(tmp_path), "train")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("rel", [_param(r) for r in TRAINABLE_CONFIGS])
def test_official_config_trains_one_epoch(rel, tmp_path, monkeypatch):
    """每个官方配置都必须能完整跑完 1 个 epoch 并产出 history.csv。"""
    from tools import train as train_tool

    config_path = os.path.join(CONFIGS_ROOT, rel)
    work_dir = str(tmp_path / "run")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tools/train.py",
            "--config",
            config_path,
            "--work-dir",
            work_dir,
            "--force",
            # 配置自带的 num_epochs 多为 1，但 classification/demo_synthetic 为 2；
            # 门禁统一压到 1 个 epoch 以保证反馈速度。
            "--cfg-options",
            "training.num_epochs=1",
        ],
    )

    # 不做 try/except：任何异常都应原样冒泡，让 pytest 给出完整 traceback。
    train_tool.main()

    run_dirs = sorted(Path(work_dir).glob("*/*"))
    assert run_dirs, f"{rel}: 未生成运行目录，work_dir={work_dir}"

    histories = list(Path(work_dir).rglob("history.csv"))
    assert histories, f"{rel}: 训练结束但未产出 history.csv（工作目录 {work_dir}）"

    configs_dumped = list(Path(work_dir).rglob("config.resolved.yaml"))
    assert configs_dumped, f"{rel}: 未落盘 config.resolved.yaml"
