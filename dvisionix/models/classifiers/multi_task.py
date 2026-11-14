# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: MultiTaskClassifier（共享骨干 + 每任务一个分类头，按样本路由）。
"""MultiTaskClassifier（共享骨干 + 每任务一个分类头，按样本路由）。

P3 的路由式多任务分类：**一次**骨干前向拿到整批特征，再按每个样本的 ``task``
取特征子集送进对应分类头。这样多任务不会把骨干计算量乘以任务数，每个头也只看
属于自己的那部分标签空间（各任务类别表彼此独立）。

用法::

    model = MultiTaskClassifier(
        backbone={"type": "mobilenetv3_backbone"},
        heads={"licence": {"num_classes": 3}, "logo": {"num_classes": 2}},
    )
    model(images)                                 # {licence: (B,3), logo: (B,2)}
    model(images, task=["licence", "logo", ...])  # {licence: (n_1,3), logo: (n_2,2)}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...errors import ConfigError
from ...registry import BACKBONES, HEADS, MODELS
from ..base import BaseModel


@MODELS.register()
@MODELS.register(name="multi_task_classifier")
class MultiTaskClassifier(BaseModel):
    """共享骨干 + 每任务一个分类头，按样本的 ``task`` 路由到对应头。

    Args:
        backbone: 骨干配置（走 ``BACKBONES.build``）。未显式给出 ``features_only``
            时按契约 P3-1 注入 ``False``，即输出整批的池化特征 ``(B, num_features)``。
        heads: ``{任务名: 分类头配置}``。每个配置会自动注入 ``in_channels``，
            且必须自带 ``num_classes``；未写 ``type`` 时默认 ``cls_head``。
        **kwargs: 兼容 ``MODELS.build`` 的额外字段（如 ``num_classes``），当前忽略。

    只读属性（契约 P3-1）：

    - ``task_names``：``tuple[str, ...]``，即 ``heads`` 的键，顺序稳定（dict 插入序）；
    - ``num_classes``：``dict[str, int]``，``{任务: head.num_classes}``。
      **不能**写成实例属性，否则会遮蔽 property，也可能让 ``training.contract``
      读到错误的值。
    """

    #: 类属性（不是实例）：供 ``training.contract`` 的能力校验读取（契约 P3-1）。
    supported_tasks = ("multi_task_classification",)

    def __init__(self, backbone: dict, heads: dict[str, dict], **kwargs: Any) -> None:
        super().__init__()
        bb_cfg = dict(backbone)
        # 多任务模型需要 (B, C) 池化特征；骨干默认按分类用法关闭多尺度输出。
        bb_cfg.setdefault("features_only", False)
        self.backbone = BACKBONES.build(bb_cfg)
        # 特征维度：内置骨干都有 num_features；兜底取多尺度输出的最后一层通道数。
        self.num_features = int(
            getattr(self.backbone, "num_features", 0)
            or getattr(self.backbone, "out_channels", [0])[-1]
        )
        if self.num_features <= 0:
            raise ConfigError(
                f"无法从骨干 {type(self.backbone).__name__} 推导特征维度："
                "它既没有 num_features，也没有 out_channels。",
                hint="请改用内置骨干（mobilenetv3_backbone / sequential_backbone 等），"
                "或在自定义骨干上提供 num_features / out_channels 属性。",
            )

        if not isinstance(heads, Mapping) or not heads:
            raise ConfigError(
                "MultiTaskClassifier 需要非空的 heads 字典（{任务名: 分类头配置}），"
                f"当前为 {heads!r}。",
                hint='示例：heads={"licence": {"type": "cls_head", "num_classes": 3}, '
                '"logo": {"num_classes": 2}}。',
            )

        self.heads = nn.ModuleDict()
        for task_name, head_cfg in heads.items():
            if not isinstance(task_name, str) or not task_name.strip():
                raise ConfigError(
                    f"heads 的任务名必须是非空字符串（当前为 {task_name!r}）。",
                    hint="任务名会作为 batch['task'] 的取值与指标前缀，请用可读的英文名。",
                )
            if "." in task_name:
                raise ConfigError(
                    f"任务名 {task_name!r} 不能包含 '.'：nn.ModuleDict 不支持这种键。",
                    hint="请把 '.' 换成 '_'（例如 licence_plate）。",
                )
            if not isinstance(head_cfg, Mapping):
                raise ConfigError(
                    f"任务 {task_name!r} 的头配置必须是字典（当前为 {type(head_cfg).__name__}）。",
                    hint='示例：{"type": "cls_head", "num_classes": 3}。',
                )
            cfg = dict(head_cfg)
            if "num_classes" not in cfg:
                raise ConfigError(
                    f"任务 {task_name!r} 的头配置缺少 num_classes。",
                    hint="多任务没有统一标签空间，每个头必须显式声明自己的类别数。",
                )
            cfg.setdefault("type", "cls_head")
            cfg.setdefault("in_channels", self.num_features)
            self.heads[task_name] = HEADS.build(cfg)

    # ------------------------------------------------------------------
    # 只读属性（契约 P3-1）
    # ------------------------------------------------------------------
    @property
    def task_names(self) -> tuple[str, ...]:
        """全部任务名（``heads`` 的键，顺序稳定）。"""
        return tuple(self.heads.keys())

    @property
    def num_classes(self) -> dict[str, int]:
        """``{任务名: 类别数}``（只读；供 ``training.contract`` 的字典类别数校验）。"""
        return {name: int(head.num_classes) for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, task: Sequence[str] | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """前向：一次骨干前向 + 按任务路由。

        Args:
            x: 输入图像 ``(B, C, H, W)``。
            task: 长度为 B 的任务名序列；``None`` 表示对整批跑全部头（推理 / 导出用）。
            **kwargs: 仅为接口兼容。P3 不把 ``labels`` 等额外参数透传给各头，
                因此 margin 头（ArcFace 等）的路由式训练不在本契约范围内。

        Returns:
            ``{任务名: logits}``。``task=None`` 时每个值是整批的 ``(B, C_task)``；
            给定 ``task`` 时是该任务子集的 ``(n_i, C_task)``，未出现的任务**不出现**
            在返回字典里（验证步的键全集由任务侧补齐，见
            ``MultiTaskClassificationTask``）。

        Raises:
            ValueError: ``task`` 长度与批大小不一致，或含未注册的任务名。
        """
        features = self._extract_features(x)
        if task is None:
            return {name: self.heads[name](features) for name in self.task_names}

        groups = self._group_by_task(task, int(features.shape[0]))
        outputs: dict[str, torch.Tensor] = {}
        for name in self.task_names:
            indices = groups.get(name)
            if not indices:
                continue
            if len(indices) == int(features.shape[0]):
                subset = features  # 整批都属于同一任务：省掉一次 index_select 拷贝
            else:
                subset = features.index_select(0, _index_tensor(indices, features.device))
            outputs[name] = self.heads[name](subset)
        return outputs

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _group_by_task(self, task: Sequence[str] | str, batch_size: int) -> dict[str, list[int]]:
        """按任务名把样本下标分组；校验长度与任务名合法性。"""
        names = [task] if isinstance(task, str) else [str(name) for name in task]
        if len(names) != batch_size:
            raise ValueError(
                f"task 长度（{len(names)}）必须等于批大小 {batch_size}："
                "每个样本都要声明自己的任务名"
            )
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            groups.setdefault(name, []).append(index)
        unknown = sorted(set(groups) - set(self.task_names))
        if unknown:
            raise ValueError(
                f"task 中出现模型没有头的任务名 {unknown}；模型的任务名集合为 "
                f"{list(self.task_names)}"
            )
        return groups

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """一次骨干前向 -> ``(B, num_features)`` 特征向量。

        兼容 ``features_only=True`` 的多尺度输出（取最后一层并全局池化）以及
        timm 的 dict 输出，避免调用方显式打开多尺度时在这里静默错位。
        """
        features = self.backbone(x)
        if isinstance(features, Mapping):
            features = list(features.values())[-1]
        elif isinstance(features, (list, tuple)):
            features = features[-1]
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"骨干 {type(self.backbone).__name__} 的输出必须是张量，"
                f"实际为 {type(features).__name__}"
            )
        if features.dim() == 4:
            features = torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
        return features


def _index_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    """把下标列表转成设备上的 long 张量（配合 ``index_select`` 保留顺序）。"""
    return torch.as_tensor(list(indices), dtype=torch.long, device=device)


__all__ = ["MultiTaskClassifier"]
