# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集的类别表（ClassTable）：顺序即 label id，拒绝重名。
"""DVX 数据集的类别表。

一个数据集的**所有标注共用一个类别表**（契约 P1-0）：类别表的顺序就是 label id，
因此：

- 传入顺序被原样保留；
- 重名会被拒绝（否则同一名字对应两个 id，映射必然出错）；
- 查询未登记的名字抛 :class:`dvisionix.errors.AnnotationError`。

本模块自身不依赖 torch / numpy / cv2，可被扫描器、切分器与工具脚本共用。

用法::

    from dvisionix.data.spec.classes import ClassTable

    table = ClassTable(["cat", "dog"])
    table.index("dog")      # 1
    table.name(0)           # "cat"
    table.has("fish")       # False
"""

from __future__ import annotations

from collections.abc import Sequence

from dvisionix.errors import AnnotationError, DatasetSpecError

__all__ = ["ClassTable"]


class ClassTable:
    """类别名 <-> label id 的双向映射。

    Args:
        names: 类别名序列，顺序即 label id；允许为空表（数据集合规性由上层检查）。

    Raises:
        DatasetSpecError: 名称不是非空字符串，或存在重复。
    """

    def __init__(self, names: Sequence[str]) -> None:
        if isinstance(names, (str, bytes, bytearray)):
            raise DatasetSpecError(
                f"类别表必须是字符串序列，不能直接传字符串: {names!r}",
                hint="请传列表，例如 ClassTable(['cat', 'dog'])",
            )
        if not isinstance(names, Sequence):
            raise DatasetSpecError(
                f"类别表必须是序列（list/tuple），当前为 {type(names).__name__}",
                hint="请传列表，例如 ClassTable(['cat', 'dog'])",
            )
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        for position, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                raise DatasetSpecError(
                    f"第 {position} 个类别名必须是非空字符串（当前为 {name!r}）",
                    hint="类别名按原样使用；空名/非字符串会让 label id 失去意义",
                )
            if name in self._index:
                raise DatasetSpecError(
                    f"类别名重复: {name!r}（第 {self._index[name]} 与第 {position} 个）",
                    hint="类别表顺序即 label id，名称必须唯一",
                )
            self._index[name] = position
            self._names.append(name)

    def __len__(self) -> int:
        """返回类别数。"""
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        """支持 ``"cat" in table``。"""
        return isinstance(name, str) and name in self._index

    def __repr__(self) -> str:
        return f"ClassTable({self._names!r})"

    def index(self, name: str) -> int:
        """返回类别名对应的 label id。

        Args:
            name: 类别名。

        Returns:
            该类别在构造时所在的下标（即 label id）。

        Raises:
            AnnotationError: 类别未登记。
        """
        try:
            return self._index[name]
        except (KeyError, TypeError) as exc:
            raise AnnotationError(
                f"未登记的类别: {name!r}",
                hint=f"已登记类别样例: {self._names[:5]}（共 {len(self._names)} 类）"
                "；请检查标注 label 拼写，或把它加入 dvx_dataset.yaml 的 classes",
            ) from exc

    def name(self, index: int) -> str:
        """返回 label id 对应的类别名。

        Args:
            index: label id（0-based）。

        Returns:
            类别名。

        Raises:
            AnnotationError: 下标越界或不是整数。
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise AnnotationError(
                f"类别下标必须是整数（当前为 {index!r}）",
                hint=f"合法范围: [0, {len(self._names) - 1}]",
            )
        if index < 0 or index >= len(self._names):
            raise AnnotationError(
                f"类别下标越界: {index}",
                hint=f"合法范围: [0, {len(self._names) - 1}]（共 {len(self._names)} 类）",
            )
        return self._names[index]

    def to_list(self) -> list[str]:
        """返回类别表副本（修改返回值不会影响 ClassTable）。"""
        return list(self._names)

    def has(self, name: str) -> bool:
        """判断类别是否已登记。"""
        return isinstance(name, str) and name in self._index
