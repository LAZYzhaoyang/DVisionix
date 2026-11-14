# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: ClassTable（类别表）单元测试：顺序、重名、未知类别。
"""ClassTable（类别表）单元测试。

覆盖契约 P1-3：

- 保持传入顺序（顺序即 label id）；
- 重名抛 ``DatasetSpecError``；
- 未登记类别抛 ``AnnotationError``；
- 空表允许构造（数据集合规性由上层检查）。
"""

import pytest

from dvisionix.data.spec.classes import ClassTable
from dvisionix.errors import AnnotationError, DatasetError, DatasetSpecError

pytestmark = pytest.mark.unit


class TestClassTableOrder:
    def test_order_is_label_id(self):
        table = ClassTable(["b", "a", "c"])
        assert len(table) == 3
        assert table.index("b") == 0
        assert table.index("a") == 1
        assert table.index("c") == 2
        assert table.name(0) == "b"
        assert table.name(1) == "a"
        assert table.name(2) == "c"

    def test_to_list_keeps_order(self):
        names = ["幻影灰", "新卡其白", "新碳晶黑", "科技灰", "纳斯达克银"]
        table = ClassTable(names)
        assert table.to_list() == names

    def test_to_list_returns_copy(self):
        table = ClassTable(["cat", "dog"])
        names = table.to_list()
        names.append("fish")
        assert table.to_list() == ["cat", "dog"]
        assert len(table) == 2

    def test_accepts_tuple_sequence(self):
        table = ClassTable(("x", "y"))
        assert table.index("y") == 1
        assert table.name(0) == "x"

    def test_repr_contains_names(self):
        assert "cat" in repr(ClassTable(["cat"]))

    def test_empty_table_allowed(self):
        table = ClassTable([])
        assert len(table) == 0
        assert table.to_list() == []
        assert not table.has("cat")


class TestClassTableLookup:
    def test_has(self):
        table = ClassTable(["cat", "dog"])
        assert table.has("cat")
        assert not table.has("fish")
        assert not table.has(1)  # type: ignore[arg-type]
        assert "dog" in table

    def test_unknown_index_raises_annotation_error(self):
        table = ClassTable(["cat", "dog"])
        with pytest.raises(AnnotationError) as excinfo:
            table.index("fish")
        assert "fish" in str(excinfo.value)
        assert excinfo.value.hint
        assert "cat" in (excinfo.value.hint or "")

    def test_unknown_index_error_is_dataset_error(self):
        table = ClassTable(["cat"])
        with pytest.raises(DatasetError):
            table.index("fish")
        with pytest.raises(AnnotationError):
            table.index("fish")

    def test_name_out_of_range(self):
        table = ClassTable(["cat", "dog"])
        with pytest.raises(AnnotationError):
            table.name(2)
        with pytest.raises(AnnotationError):
            table.name(-1)

    def test_name_requires_int(self):
        table = ClassTable(["cat"])
        with pytest.raises(AnnotationError):
            table.name("0")  # type: ignore[arg-type]
        with pytest.raises(AnnotationError):
            table.name(True)  # type: ignore[arg-type]


class TestClassTableConstructionErrors:
    def test_duplicate_raises_dataset_spec_error(self):
        with pytest.raises(DatasetSpecError) as excinfo:
            ClassTable(["cat", "dog", "cat"])
        assert "重复" in str(excinfo.value)
        assert excinfo.value.hint

    def test_duplicate_not_adjacent(self):
        with pytest.raises(DatasetSpecError):
            ClassTable(["a", "b", "c", "a"])

    def test_empty_name_rejected(self):
        with pytest.raises(DatasetSpecError):
            ClassTable(["cat", ""])
        with pytest.raises(DatasetSpecError):
            ClassTable(["cat", "   "])

    def test_non_string_name_rejected(self):
        with pytest.raises(DatasetSpecError):
            ClassTable(["cat", 1])  # type: ignore[list-item]

    def test_string_argument_rejected(self):
        with pytest.raises(DatasetSpecError):
            ClassTable("cat")  # type: ignore[arg-type]

    def test_non_sequence_rejected(self):
        with pytest.raises(DatasetSpecError):
            ClassTable(42)  # type: ignore[arg-type]

    def test_error_does_not_mutate_input(self):
        names = ["cat", "cat"]
        with pytest.raises(DatasetSpecError):
            ClassTable(names)
        assert names == ["cat", "cat"]
