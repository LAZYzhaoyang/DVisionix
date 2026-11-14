# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集规格草稿生成器（契约 P1-8）。
"""DVX 数据集 ``dvx_dataset.yaml`` 草稿生成器（契约 P1-8）。

只做"只读扫描 + 生成草稿"，**绝不**移动/改名/删除任何图片或 JSON：

- 自动推断 ``layout``：存在与图片同名的 ``<stem>.json`` -> ``labelme``，
  否则 -> ``folder_per_class``；显式 ``--layout`` 优先（与推断不一致时提示）。
- ``folder_per_class``：类别 = 一级子目录名（排除 ``train_data`` / ``train`` / ``val``
  等切片目录）；多任务时用 ``--task-dir`` 指定任务目录，类别取该目录的一级子目录。
- ``labelme``：类别 = 所有 ``shapes[*].label`` 的并集（排序后写入），默认忽略
  ``__ignore__`` / ``__background__``。
- ``--out`` 未给时把 YAML 草稿打印到 stdout；给了才写文件，已存在且无 ``--force``
  时报错。**不会**自动写入数据目录。
- 输出末尾提示"请人工确认 classes 顺序（顺序即 label id）"。

用法::

    python tools/init_dataset.py --root project/data/det --data-type detection
    python tools/init_dataset.py --root project/data/det --data-type detection \
        --out project/data/det/dvx_dataset.yaml
    python tools/init_dataset.py --root project/data/multicls \
        --data-type multi_task_classification \
        --task-dir T15C_1CCD6_licence --task-dir T15C_1CCD6_logo \
        --out project/data/multicls/dvx_dataset.yaml

退出码：成功 0；参数/目录/标注问题 1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Iterable, Sequence

import yaml

# 让 `python tools/init_dataset.py` 在仓库根目录之外也能 import 到本仓库的 dvisionix
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvisionix.data.spec.schema import (  # noqa: E402
    DATA_TYPE_LAYOUTS,
    DATA_TYPES,
    LAYOUTS,
)
from dvisionix.errors import DatasetSpecError, DVisionixError  # noqa: E402

#: 参与草稿统计/类别推断的图片扩展名（与 ParseConfig 默认值一致）
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
#: 默认忽略的 labelme 标签（与 ParseConfig 默认值一致）
DEFAULT_IGNORE_LABELS = frozenset({"__ignore__", "__background__"})
#: 视为"切片目录"而非类别目录的目录名（避免把 train_data 当成类别）
SPLIT_DIR_NAMES = frozenset(
    {
        "train",
        "val",
        "valid",
        "validation",
        "test",
        "train_data",
        "val_data",
        "test_data",
        "train_dataset",
        "val_dataset",
        "test_dataset",
    }
)
#: 草稿末尾的人工确认提示（契约 P1-8 要求）
REMINDER = "请人工确认 classes 顺序（顺序即 label id）"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="扫描数据目录，生成 DVX 数据集规格（dvx_dataset.yaml）草稿。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="数据集根目录（只读扫描）")
    parser.add_argument(
        "--data-type",
        required=True,
        choices=DATA_TYPES,
        help="数据类型（决定 layout 合法组合与默认 splits）",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS,
        default=None,
        help="目录布局；缺省时按是否存在同名 JSON 自动推断",
    )
    parser.add_argument(
        "--task-dir",
        action="append",
        default=None,
        metavar="REL_DIR",
        help="多任务分类的任务子目录（相对 --root，可重复）；缺省时自动发现一级任务目录",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="YAML 草稿输出路径；缺省时打印到 stdout（不会写入数据目录）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖 --out 指定的已存在文件",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回进程退出码。"""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except DVisionixError as exc:
        print(f"[init_dataset] 失败：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"[init_dataset] 失败：{exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    """执行扫描与草稿生成（失败统一抛 DVisionixError 系列）。"""
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        raise DatasetSpecError(
            f"数据根目录不存在或不是目录: {root}",
            hint="请把 --root 指向实际的数据集目录（例如 project/data/det）",
        )

    inferred_layout, image_count = _infer_layout(root)
    layout = args.layout or inferred_layout
    if args.layout is not None and args.layout != inferred_layout:
        _warn(
            f"显式 layout={args.layout!r} 与自动推断 {inferred_layout!r} 不一致，"
            "按显式值生成草稿；请确认目录内容"
        )
    allowed_layouts = DATA_TYPE_LAYOUTS[args.data_type]
    if layout not in allowed_layouts:
        raise DatasetSpecError(
            f"data_type={args.data_type!r} 与 layout={layout!r} 组合非法",
            hint=f"{args.data_type} 只允许 layout: {list(allowed_layouts)}；"
            "请检查 --data-type，或显式传入合法的 --layout",
        )

    out_path = None
    if args.out is not None:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        if os.path.exists(out_path) and not args.force:
            raise DatasetSpecError(
                f"输出文件已存在: {out_path}",
                hint="如需覆盖请加 --force，或换一个 --out 路径",
            )
        root_value = _root_relative_to(root, out_path)
    else:
        root_value = root

    document: dict[str, Any] = {
        "version": 1,
        "data_type": args.data_type,
        "layout": layout,
        "root": root_value,
    }

    if args.data_type == "multi_task_classification":
        tasks = _discover_tasks(root, args.task_dir)
        document["tasks"] = [
            {"name": name, "path": rel_path, "classes": classes}
            for name, rel_path, classes in tasks
        ]
        _print_task_stats(tasks)
    else:
        if args.task_dir:
            raise DatasetSpecError(
                "--task-dir 只用于 multi_task_classification",
                hint=f"当前 --data-type={args.data_type}，请去掉 --task-dir",
            )
        if layout == "folder_per_class":
            classes = _discover_classes(root)
            document["classes"] = classes
            _print_class_stats("folder_per_class", classes, image_count)
        else:
            classes, json_count, shape_types = _labelme_labels(root)
            document["classes"] = classes
            _print_class_stats("labelme", classes, image_count)
            _warn_labelme_shape_types(args.data_type, shape_types, json_count, image_count)

    if args.data_type == "segmentation":
        document["seg_mode"] = "semantic"
        document["ignore_index"] = 255
    document["splits"] = _default_splits(args.data_type)

    text = _render_yaml(document)
    if out_path is None:
        print(text, end="")
    else:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[init_dataset] 已写入 spec 草稿: {out_path}")
        print(REMINDER)
    print(f"[init_dataset] {REMINDER}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# 目录/标注扫描
# ---------------------------------------------------------------------------


def _infer_layout(root: str) -> tuple[str, int]:
    """按"图片是否有同名 JSON"推断布局，并返回图片总数。"""
    image_count = 0
    has_json = False
    for image_path in _iter_image_files(root):
        image_count += 1
        if not has_json and os.path.isfile(os.path.splitext(image_path)[0] + ".json"):
            has_json = True
    if image_count == 0:
        raise DatasetSpecError(
            f"目录中没有可用于推断的图片: {root}",
            hint=f"支持的扩展名: {sorted(IMAGE_EXTENSIONS)}；请检查 --root",
        )
    return ("labelme" if has_json else "folder_per_class"), image_count


def _discover_classes(base: str) -> list[str]:
    """推断 folder_per_class 的类别名（排序后）。

    优先取"直接子目录中直接/递归含图片且名字不是切片目录"的那些；一个都没有时
    （例如只有 ``train_data/train/<class>`` 而没有扁平类目录）退化为"任意直接包含
    图片的目录名"，从而兼容"只保留切片目录"的布局。
    """
    primary: list[str] = []
    split_dirs: list[str] = []
    for path in _list_subdirs(base):
        name = os.path.basename(path)
        if name in SPLIT_DIR_NAMES:
            split_dirs.append(path)
            continue
        if _contains_image(path):
            primary.append(name)
    if primary:
        return sorted(primary)

    # 退化路径：从整个子树里收集"直接包含图片的目录名"（切片目录本身不计）
    names: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(base):
        if dirpath == base or os.path.basename(dirpath) in SPLIT_DIR_NAMES:
            continue
        if any(_is_image_name(name) for name in filenames):
            names.add(os.path.basename(dirpath))
    if names:
        _warn(
            f"未在 {base} 的一级子目录中找到类别目录，已从切片子目录推断类别 {sorted(names)}；"
            "请人工确认"
        )
        return sorted(names)

    got_split_dirs = [os.path.basename(path) for path in split_dirs]
    raise DatasetSpecError(
        f"无法从目录推断类别: {base}",
        hint="folder_per_class 要求 <base>/<class>/*.jpg；"
        f"当前一级子目录: {[os.path.basename(path) for path in _list_subdirs(base)]}"
        f"（切片目录: {got_split_dirs}）",
    )


def _discover_tasks(root: str, task_dirs: Sequence[str] | None) -> list[tuple[str, str, list[str]]]:
    """返回多任务分类的 ``(task_name, rel_path, classes)`` 列表。"""
    if task_dirs:
        tasks: list[tuple[str, str, list[str]]] = []
        seen: set[str] = set()
        for raw in task_dirs:
            rel_path = _normalize_relative_dir(raw, "--task-dir")
            base = os.path.join(root, rel_path.replace("/", os.sep))
            if not os.path.isdir(base):
                raise DatasetSpecError(
                    f"--task-dir 指向的目录不存在: {base}",
                    hint="--task-dir 必须相对 --root，例如 T15C_1CCD6_licence",
                )
            name = os.path.basename(base)
            if name in seen:
                raise DatasetSpecError(
                    f"任务目录名重复: {name!r}",
                    hint="task.name 必须唯一；请用不同的一级目录区分任务",
                )
            seen.add(name)
            tasks.append((name, rel_path, _discover_classes(base)))
        return tasks

    tasks = []
    for path in _list_subdirs(root):
        name = os.path.basename(path)
        if name in SPLIT_DIR_NAMES or not _contains_image(path):
            continue
        try:
            classes = _discover_classes(path)
        except DatasetSpecError:
            continue
        tasks.append((name, name, classes))
    if not tasks:
        raise DatasetSpecError(
            f"未在 {root} 下发现任何任务目录",
            hint="多任务分类请用 --task-dir 指定任务目录（可重复），"
            "或保证每个任务目录下存在 <class>/*.jpg",
        )
    return tasks


def _labelme_labels(root: str) -> tuple[list[str], int, Counter]:
    """收集全部 labelme shape label（排序后）与 JSON 数、shape_type 计数。"""
    labels: set[str] = set()
    json_count = 0
    shape_types: Counter = Counter()
    for json_path in _iter_json_files(root):
        json_count += 1
        data = _load_json(json_path)
        shapes = data.get("shapes") or []
        if not isinstance(shapes, list):
            raise DatasetSpecError(
                f"labelme JSON 的 shapes 必须是列表: {json_path}"
                f"（当前为 {type(shapes).__name__}）",
                hint="请用 labelme 重新保存该文件；没有形状时应写 shapes: []",
            )
        for index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                raise DatasetSpecError(
                    f"labelme JSON 的 shapes[{index}] 必须是映射: {json_path}",
                    hint="每个形状应形如 {label: 'cat', shape_type: 'polygon', points: [...]}",
                )
            label = shape.get("label")
            if not isinstance(label, str) or not label.strip():
                raise DatasetSpecError(
                    f"labelme JSON 的 shapes[{index}] 缺少合法 label: {json_path}"
                    f"（当前为 {label!r}）",
                    hint="label 必须是非空字符串",
                )
            shape_type = shape.get("shape_type")
            shape_types[str(shape_type) if isinstance(shape_type, str) else "<missing>"] += 1
            if label in DEFAULT_IGNORE_LABELS:
                continue
            labels.add(label)

    if not labels:
        raise DatasetSpecError(
            f"未在 labelme 标注中发现任何类别: {root}",
            hint="请确认目录下有与图片同名的 <stem>.json，且 shapes 的 label 非空",
        )
    return sorted(labels), json_count, shape_types


def _load_json(json_path: str) -> dict[str, Any]:
    """读取并校验 labelme JSON 顶层结构。"""
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise DatasetSpecError(
            f"无法读取 labelme 标注: {json_path}（{exc}）",
            hint="请检查文件权限与编码（要求 UTF-8）",
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetSpecError(
            f"labelme JSON 解析失败: {json_path}（{exc.msg}，第 {exc.lineno} 行）",
            hint="请用 labelme 重新保存，或用 jq 校验 JSON 语法",
        ) from exc
    if not isinstance(data, dict):
        raise DatasetSpecError(
            f"labelme JSON 顶层必须是映射: {json_path}" f"（当前为 {type(data).__name__}）",
            hint="labelme 导出文件顶层是对象，应包含 shapes / imageWidth / imageHeight",
        )
    return data


# ---------------------------------------------------------------------------
# 草稿渲染
# ---------------------------------------------------------------------------


def _default_splits(data_type: str) -> dict[str, Any]:
    """按数据类型给出切分默认值（与 schema.py 的默认值保持一致）。"""
    splits: dict[str, Any] = {
        "mode": "ratio",
        "ratio": {"train": 0.8, "val": 0.2},
        "seed": 42,
    }
    splits["stratify"] = (
        "class" if data_type != "detection" and data_type != "segmentation" else "none"
    )
    return splits


def _render_yaml(document: dict[str, Any]) -> str:
    """把规格字典渲染为带说明注释的 YAML 文本。"""
    body = yaml.safe_dump(
        document,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=None,
    )
    lines = [
        "# DVX 数据集规格草稿（由 tools/init_dataset.py 生成；已扫描目录，未改动任何文件）",
        body.rstrip("\n"),
        f"# {REMINDER}",
        "",
    ]
    return "\n".join(lines)


def _root_relative_to(root: str, out_path: str) -> str:
    """计算相对 ``--out`` 所在目录的 root（posix 风格），便于 spec 随手移动。"""
    relative = os.path.relpath(root, os.path.dirname(out_path) or os.curdir)
    return relative.replace(os.sep, "/") or "."


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _iter_image_files(base: str) -> Iterable[str]:
    """确定性遍历根目录下的图片文件（跳过隐藏目录/文件）。"""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if _is_image_name(name):
                yield os.path.join(dirpath, name)


def _iter_json_files(base: str) -> Iterable[str]:
    """确定性遍历根目录下的 JSON 文件（跳过隐藏目录/文件）。"""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if not name.startswith(".") and name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)


def _is_image_name(name: str) -> bool:
    """判断文件名是否命中图片扩展名（小写比较）。"""
    return not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _list_subdirs(base: str) -> list[str]:
    """返回 ``base`` 的直接子目录（跳过隐藏目录，按名字排序）。"""
    try:
        entries = sorted(os.scandir(base), key=lambda entry: entry.name)
    except OSError as exc:
        raise DatasetSpecError(
            f"无法读取目录: {base}（{exc}）",
            hint="请检查目录权限，或确认它是一个目录",
        ) from exc
    return [
        entry.path
        for entry in entries
        if entry.name and not entry.name.startswith(".") and entry.is_dir()
    ]


def _contains_image(base: str) -> bool:
    """判断 ``base`` 子树下是否存在至少一张图片。"""
    for _dirpath, _dirnames, filenames in os.walk(base):
        if any(_is_image_name(name) for name in filenames):
            return True
    return False


def _normalize_relative_dir(value: str, where: str) -> str:
    """把命令行传入的相对目录规范化为 posix 风格并拒绝绝对路径/``..``。"""
    text = str(value).replace("\\", "/").strip().strip("/")
    if not text or text.startswith("/"):
        raise DatasetSpecError(
            f"{where} 必须是非空的相对目录（当前为 {value!r}）",
            hint="例如 T15C_1CCD6_licence",
        )
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise DatasetSpecError(
            f"{where} 不能包含 '..' 或为空（当前为 {value!r}）",
            hint="请使用相对 --root 的子目录，例如 T15C_1CCD6_licence",
        )
    return "/".join(parts)


def _print_class_stats(source: str, classes: Sequence[str], image_count: int) -> None:
    """把类别/样本摘要打印到 stderr（stdout 保持纯 YAML）。"""
    print(
        f"[init_dataset] layout={source}，图片 {image_count} 张，类别 {len(classes)} 个: {list(classes)}",
        file=sys.stderr,
    )


def _print_task_stats(tasks: Sequence[tuple[str, str, list[str]]]) -> None:
    """把多任务摘要打印到 stderr。"""
    for name, rel_path, classes in tasks:
        print(
            f"[init_dataset] 任务 {name!r}（{rel_path}）：类别 {len(classes)} 个: {list(classes)}",
            file=sys.stderr,
        )


def _warn_labelme_shape_types(
    data_type: str, shape_types: Counter, json_count: int, image_count: int
) -> None:
    """提示图像/JSON 数量不一致，以及与本数据类型不匹配的形状。"""
    if json_count != image_count:
        _warn(f"图片 {image_count} 张，JSON {json_count} 个，数量不一致；扫描时缺 JSON 会报错")
    expected = "rectangle" if data_type == "detection" else "polygon"
    unexpected = {name: count for name, count in shape_types.items() if name != expected}
    if unexpected:
        _warn(
            f"data_type={data_type} 只保留 {expected} 形状；"
            f"以下形状扫描时会作为 skipped_shapes 跳过: {unexpected}"
        )


def _warn(message: str) -> None:
    """向 stderr 打印一行提示（不污染 stdout 的 YAML）。"""
    print(f"[init_dataset] 提示: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
