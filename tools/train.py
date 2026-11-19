# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 配置驱动的统一训练入口（v0.3.0）。
"""配置驱动的统一训练入口（v0.3.0）。

用法::

    python tools/train.py --config configs/classification/demo_synthetic.yaml
    python tools/train.py --config xxx.yaml --cfg-options training.learning_rate=0.01
    python tools/train.py --config xxx.yaml --resume auto            # 续最近一次任务
    python tools/train.py --config xxx.yaml --work-dir /path/to/run  # 指定工作目录
    torchrun --nproc_per_node=2 tools/train.py --config xxx.yaml --devices 0,1

配置需包含 task_type / model / data / training 等字段，详见 CodePlan.md。
训练产物（日志 / TensorBoard / 检查点 / 最终配置）全部落在 work_dir（默认在代码库外）。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from dvisionix.config import Config
from dvisionix.data import CustomDataset, DVXDataset, build_dataset
from dvisionix.errors import ConfigError
from dvisionix.models import build_model
from dvisionix.training import (
    build_task,
    build_trainer,
    dump_config,
    find_checkpoint,
    load_backbone,
    resolve_work_dir,
)
from dvisionix.utils import get_logger, set_seed

# 任务类型 -> Task 类名。键集合必须与 dvisionix.config.schema.TASK_TYPES 完全一致；
# 该不变量由 tests/test_config/test_schema.py::test_task_mapping_covers_all_task_types 守护。
# （原先此处另有一份 _TASK_TYPES 常量，从未被引用，已删除以免与权威来源漂移。）
_TASK_MAPPING = {
    "classification": "ClassificationTask",
    "detection": "DetectionTask",
    "segmentation": "SegmentationTask",
    # 路由式多任务分类（P3）：每个样本按自身 task 路由到对应分类头
    "multi_task_classification": "MultiTaskClassificationTask",
    "simclr": "SimCLRTask",
}

#: 多任务分类的任务类型名。该任务没有全局 ``model.num_classes``：
#: 类别数由每个 head 自带（模型 ``num_classes`` 是 dict）。
_MULTI_TASK_CLASSIFICATION = "multi_task_classification"


def build_transforms(task, image_size, train):
    """向任务索取预设变换流水线（契约 P2-2：任务承担任务知识）。

    训练入口不再按 ``task_type`` 硬编码 if/else 分派，而是统一委托
    ``task.build_transforms(train, image_size)``。

    Args:
        task: 任务实例。
        image_size: 目标图像边长。
        train: True 构建训练变换，False 构建验证变换。

    Raises:
        ConfigError: 任务未提供预设变换（``build_transforms`` 返回 None）。
    """
    transforms = task.build_transforms(train, image_size)
    if transforms is None:
        raise ConfigError(
            f"任务 {type(task).__name__} 未提供预设变换（build_transforms 返回 None）",
            hint="该任务未提供预设变换；请在任务类中实现 build_transforms(train, image_size)，"
            "或改用已提供预设变换的内置任务",
        )
    return transforms


def build_synthetic_dataset(
    task_type, num_samples, num_classes, image_size, transforms, cache_dir=None, split="train"
):
    """生成内存合成数据集（便于无网络环境快速验证）。

    Args:
        split: 数据集划分标识（``train`` / ``val``）。**必须区分** ——
            v1.0.0 的 train 与 val 共用同一目录和 ``img_%04d.png`` 命名，
            于是验证集的前 N 张就是训练集的前 N 张，验证指标天然含泄漏
            （CodePlan 7.1.2 D13）。
    """
    import cv2

    tmp_dir = cache_dir or os.path.join(".cache", "synthetic", task_type)
    os.makedirs(tmp_dir, exist_ok=True)
    # 稳定的划分种子；不能用内置 hash()（对 str 按进程随机化）
    split_seed = sum(ord(c) for c in str(split)) * 100003
    samples = []
    for i in range(num_samples):
        rng = np.random.default_rng([split_seed, i])
        img = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        path = os.path.join(tmp_dir, f"{split}_img_{i:04d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, img)
        if task_type == "classification":
            samples.append({"image": path, "label": i % num_classes})
        elif task_type == "detection":
            half = max(2, image_size // 2)
            x1, y1 = (int(v) for v in rng.integers(0, half, 2))
            x2 = x1 + int(rng.integers(1, half))
            y2 = y1 + int(rng.integers(1, half))
            samples.append(
                {
                    "image": path,
                    "boxes": np.array(
                        [[float(x1), float(y1), float(x2), float(y2)]], dtype=np.float32
                    ),
                    "labels": np.array([i % num_classes], dtype=np.int64),
                }
            )
        elif task_type == "segmentation":
            mask = (rng.random((image_size, image_size)) * num_classes).astype(np.uint8)
            mask_path = os.path.join(tmp_dir, f"{split}_mask_{i:04d}.png")
            cv2.imwrite(mask_path, mask)
            samples.append({"image": path, "mask": mask_path})
        elif task_type == "simclr":
            # 自监督：标签由 SimCLRTransforms 对同一图像生成的双视角（image1 / image2）提供，
            # Sample 只需给出 image 即可，不需要 label / boxes / mask。
            samples.append({"image": path})
        else:
            raise ValueError(
                f"合成数据不支持 task_type={task_type!r}；"
                f"可选: {sorted(_TASK_MAPPING)}。新增任务类型时必须同时在此登记。"
            )
    return CustomDataset(samples=samples, task_type=task_type, transforms=transforms)


def build_data(cfg, task, work_dir=None):
    """根据配置构建 (train_loader, val_loader)。

    数据源三选一：

    - ``data.dataset: dvx`` + ``data.spec``：按 ``dvx_dataset.yaml`` 构建
      ``DVXDataset``（分类 / 检测 / 分割 / 多任务分类）；
    - ``custom`` / ``synthetic``：内存合成数据（无网络环境快速验证）；
    - 其它名称：走 ``build_dataset`` 的公开数据集预设。

    预设变换统一来自 ``task.build_transforms``（契约 P2-2），不再由本入口按
    ``task_type`` 硬编码。

    Args:
        cfg: 全局配置。
        task: 任务实例（提供 ``build_transforms``）。
        work_dir: 工作目录；合成数据缓存写到 work_dir/.cache/synthetic（隔离代码库）。

    Raises:
        ConfigError: ``data.dataset: dvx`` 缺少 ``data.spec``，或任务未提供预设变换。
    """
    task_type = cfg.task_type
    image_size = cfg.data.image_size

    data_cfg = cfg.data
    dataset_name = data_cfg.get("dataset", "custom")
    spec_path = None
    if dataset_name == "dvx":
        spec_path = data_cfg.get("spec")
        if not spec_path:
            raise ConfigError(
                "data.dataset 为 dvx 时必须提供 data.spec（dvx_dataset.yaml 路径）",
                hint="例如 data.spec: ./data/cls/dvx_dataset.yaml",
            )

    train_tf = build_transforms(task, image_size, train=True)
    val_tf = build_transforms(task, image_size, train=False)

    if dataset_name == "dvx":
        train_ds = DVXDataset(spec_path, split="train", transforms=train_tf)
        val_ds = DVXDataset(spec_path, split="val", transforms=val_tf)
    elif dataset_name in ("custom", "synthetic") or data_cfg.get("synthetic", False):
        n_train = data_cfg.get("num_samples", 64)
        n_val = data_cfg.get("val_samples", 16)
        # 只有合成/自定义数据源需要全局类别数；多任务分类没有全局类别数，
        # 因此不得在函数开头无条件读取 cfg.model.num_classes（P3 契约 §4.4）。
        num_classes = cfg.model.num_classes
        cache_dir = os.path.join(work_dir, ".cache", "synthetic", task_type) if work_dir else None
        # train / val 必须用不同 split，否则二者共用文件名序列、验证集成为训练集子集
        train_ds = build_synthetic_dataset(
            task_type, n_train, num_classes, image_size, train_tf, cache_dir, split="train"
        )
        val_ds = build_synthetic_dataset(
            task_type, n_val, num_classes, image_size, val_tf, cache_dir, split="val"
        )
    else:
        root = data_cfg.get("root", "./data")
        train_kwargs = dict(
            root=root, train=True, transforms=train_tf, download=data_cfg.get("download", False)
        )
        val_kwargs = dict(
            root=root, train=False, transforms=val_tf, download=data_cfg.get("download", False)
        )
        for extra_key in ("year", "image_set", "split"):
            if extra_key in data_cfg:
                train_kwargs[extra_key] = data_cfg[extra_key]
                val_kwargs[extra_key] = data_cfg[extra_key]
        train_ds = build_dataset({"type": dataset_name, **train_kwargs})
        val_ds = build_dataset({"type": dataset_name, **val_kwargs})

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 0)
    training_cfg = cfg.training
    # 数据加载性能开关（配置驱动，默认值与 torch 保持一致）
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", False))
        if training_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=getattr(train_ds, "collate_fn", None),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=getattr(val_ds, "collate_fn", None),
        **loader_kwargs,
    )
    return train_loader, val_loader


def build_task_cfg(cfg):
    """从配置组装 Task 构建参数（optimizer/scheduler/loss/metrics 全部配置驱动）。"""
    task_cfg = dict(cfg.get("task", {}) or {})
    if "type" not in task_cfg:
        task_cfg["type"] = _TASK_MAPPING[cfg.task_type]
    # 多任务分类没有全局类别数：模型 num_classes 是 {task: C} 的 dict，且由 heads 自带，
    # 因此不向任务注入 num_classes（P3 契约 §4.4）。其余任务保持原行为。
    if cfg.task_type != _MULTI_TASK_CLASSIFICATION:
        task_cfg.setdefault("num_classes", cfg.model.num_classes)

    training = cfg.get("training", {}) or {}
    optimizer_cfg = dict(training.get("optimizer", {}) or {})
    optimizer_cfg.setdefault("lr", training.get("learning_rate", 1e-3))
    optimizer_cfg.setdefault("weight_decay", training.get("weight_decay", 1e-4))
    task_cfg.setdefault("optimizer_cfg", optimizer_cfg)

    scheduler_cfg = dict(training.get("scheduler", {}) or {})
    task_cfg.setdefault("scheduler_cfg", scheduler_cfg)

    loss = cfg.get("loss")
    if loss is not None:
        loss = loss.to_dict() if isinstance(loss, Config) else dict(loss)
        if cfg.task_type == "detection" and "num_classes" not in loss:
            loss["num_classes"] = cfg.model.num_classes
        task_cfg.setdefault("loss", loss)

    metrics = cfg.get("metrics")
    if metrics is not None:
        metrics = metrics.to_dict() if isinstance(metrics, Config) else dict(metrics)
        task_cfg.setdefault("metrics", metrics)

    return task_cfg


def parse_devices(value):
    """'0,1' -> [0, 1]。"""
    if value is None:
        return None
    return [int(x) for x in str(value).split(",") if x.strip() != ""]


def export_best_onnx(cfg, work_dir: str, logger=None):
    """把最优 checkpoint（work_dir/checkpoints/best.pt）导出为 ONNX（best-effort）。

    仅当 ``training.export_best_onnx`` 为 true 时调用；模型不可 trace 时告警降级，
    不影响训练主流程。
    """
    best_pt = os.path.join(work_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_pt):
        if logger:
            logger.warning(f"export_best_onnx 已开启但未找到 {best_pt}")
        return None
    try:
        import torch

        from dvisionix.export import build_exporter

        model = build_model(cfg.model.to_dict())
        state = torch.load(best_pt, map_location="cpu", weights_only=False)
        if state.get("model_state_dict"):
            model.load_state_dict(state["model_state_dict"])
        in_channels = cfg.model.get("in_channels", 3)
        image_size = cfg.data.get("image_size", 32)
        exporter = build_exporter(
            "onnx",
            model,
            input_shape=(in_channels, image_size, image_size),
            device="cpu",
            task_type=cfg.get("task_type"),
        )
        out_path = os.path.join(work_dir, "best.onnx")
        result = exporter.export(out_path, dynamic_batch=True)
        result.save(f"{out_path}.json")
        if logger:
            logger.info(f"Best model exported to ONNX: {result.path}")
        return result.path
    except Exception as exc:  # pragma: no cover - 依赖具体模型可 trace 性
        if logger:
            logger.warning(f"导出 best.onnx 失败（已忽略）：{exc}")
        return None


def main():
    """配置驱动训练入口：解析参数 -> 构建数据/模型/任务/Trainer -> 训练 -> 可选 ONNX 导出。"""
    parser = argparse.ArgumentParser(description="DVisionix config-driven training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="override config, e.g. training.learning_rate=0.01",
    )
    parser.add_argument(
        "--resume", default=None, help="resume mode: auto / latest / <checkpoint path>"
    )
    parser.add_argument(
        "--work-dir", default=None, help="explicit work dir (default: ~/dvisionix_runs/<exp>/<ts>)"
    )
    parser.add_argument("--devices", default=None, help="devices for DDP, e.g. '0,1'")
    parser.add_argument("--strategy", default=None, help="auto / ddp / none")
    parser.add_argument("--force", action="store_true", help="force fresh run (ignore resume)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.update_from_cli(args.cfg_options)
    required_keys = ["task_type", "model.num_classes", "training.num_epochs"]
    if cfg.get("task_type") == _MULTI_TASK_CLASSIFICATION:
        # 多任务分类的类别数分散在各 head 中，不存在全局 model.num_classes
        required_keys.remove("model.num_classes")
    cfg.validate(required_keys)
    schema_warnings = cfg.validate_schema(cfg.task_type)

    resume = None if args.force else (args.resume or cfg.get("resume", False))
    if resume is True:
        # `resume: true` 是常见写法，统一归一为 "auto"（否则会被当成路径传给 torch.load）
        resume = "auto"
    work_dir = resolve_work_dir(cfg, cli_work_dir=args.work_dir, resume=resume)
    resume_path = find_checkpoint(work_dir, resume)
    os.makedirs(work_dir, exist_ok=True)
    dump_config(cfg, os.path.join(work_dir, "config.resolved.yaml"))

    seed = cfg.training.get("seed", 42)
    set_seed(seed)
    logger = get_logger("dvisionix.train", level="info", log_dir=os.path.join(work_dir, "logs"))
    logger.info(f"Config: {args.config}")
    logger.info(f"Task: {cfg.task_type}, model: {cfg.model.get('type') or cfg.model.get('name')}")
    logger.info(f"Work dir: {work_dir}")
    if resume_path:
        logger.info(f"Resume from: {resume_path}")
    for w in schema_warnings:
        logger.warning(w)

    # 构造顺序（契约 P2-5）：config -> task -> transforms(task) -> data -> model -> trainer。
    # 先建任务，数据装配才能向任务索取预设变换（P2-2），入口不再硬编码任务分派。
    task = build_task(build_task_cfg(cfg))

    train_loader, val_loader = build_data(cfg, task, work_dir=work_dir)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg.model.to_dict())
    logger.info(f"Model params: {model.count_parameters():,}")

    pretrained_backbone = cfg.model.get("pretrained_backbone") or cfg.training.get(
        "pretrained_backbone"
    )
    if pretrained_backbone:
        load_backbone(model, pretrained_backbone)
        logger.info(f"Backbone pretrained weights loaded from: {pretrained_backbone}")

    trainer = build_trainer(
        cfg,
        task,
        train_loader,
        val_loader,
        work_dir=work_dir,
        resume_from=resume_path,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
    )
    trainer.fit(model)

    if cfg.training.get("export_best_onnx", False):
        export_best_onnx(cfg, work_dir, logger=logger)


if __name__ == "__main__":
    main()
