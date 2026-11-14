# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: folder_per_class 布局扫描器（<root>[/<task.path>][/<subdir>]/<class>/*.jpg）。
"""``folder_per_class`` 布局的扫描器（契约 P1-6）。

布局::

    <root>[/<task.path>][/<subdir>]/<class>/*.jpg

扫描规则（契约 P1-0 / P1-4 / P1-6）：

- **只有目录名出现在类别表中的目录才算类别目录**。``train_data`` / ``images`` 这类与
  类别无关的目录不会被误当成类别名，它们本身不产生任何样本。
- 一个图片文件归属"**最近的类别祖先目录**"：从它所在目录向上找，第一个目录名在类别表
  里出现的目录即它的类别。因此 ``<class>/x.jpg`` 与 ``<class>/sub/x.jpg`` 都属于该类，
  而 ``train_data/train/<class>/x.jpg`` 仍归属 ``<class>``（契约 P1-10 的 multicls 布局）。
- ``spec.parse.recursive=False`` 时只看 ``<base>/<class>/<file>`` 这一层，不进入任何
  子目录；``True``（默认）时递归整个子树。
- 隐藏目录与隐藏文件（``.`` 开头）一律跳过；``dvx_dataset.yaml`` 不是图片，自然不参与。
- 图片扩展名按 ``spec.parse.image_ext`` **小写比较**（``.JPG`` 能命中 ``.jpg``）。
- 同一目录下同一 stem 的 ``x.jpg`` 与 ``x.png`` 并存 → ``AnnotationError``：
  无法判定哪个文件才是样本，且同一张图重复计入会让类别统计与切分失真。
- 输出按 ``rel_path`` 确定性升序排列（契约 P1-4 的确定性要求）。

本模块只读磁盘，不移动/改名任何文件（契约 P1-0 的非破坏性要求）。

用法::

    from dvisionix.data.parsers.folder import scan_folder_per_class
    from dvisionix.data.spec.schema import load_spec

    spec = load_spec("project/data/cls/dvx_dataset.yaml")
    refs = scan_folder_per_class(spec.root, spec)
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping

from dvisionix.data.spec.classes import ClassTable
from dvisionix.data.spec.scan import SampleRef
from dvisionix.data.spec.schema import DVXDatasetSpec, TaskSpec
from dvisionix.errors import AnnotationError, DatasetSpecError

__all__ = ["scan_folder_per_class"]


def scan_folder_per_class(
    root: str,
    spec: DVXDatasetSpec,
    task: TaskSpec | None = None,
    subdir: str | None = None,
) -> list[SampleRef]:
    """扫描 folder_per_class 布局，返回按 ``rel_path`` 升序的样本列表。

    Args:
        root: 数据集根目录（通常是 ``spec.root``；也可指向其子目录）。
        spec: 数据集规格，提供类别表（单任务用 ``spec.classes``，多任务用 ``task.classes``）、
            图片扩展名（``spec.parse.image_ext``）与递归开关（``spec.parse.recursive``）。
        task: 多任务分类的子任务；给定时扫描 ``<root>/<task.path>`` 并用该任务的类别表，
            样本的 ``SampleRef.task`` 记为 ``task.name``。
        subdir: 可选的额外子目录层（如 ``train_data``），相对 ``root`` 下的任务目录；
            用于把扫描限制在切片目录内，避免与扁平类目录重复计入（契约 P1-10）。

    Returns:
        样本列表，``rel_path`` 相对 ``spec.root``（posix 风格，含 ``task.path`` 前缀），
        按 ``rel_path`` 升序；每个样本带 ``label``（类别 id）与
        ``meta["class_name"]``（类别名，便于排查）。

    Raises:
        AnnotationError: 扫描根目录不存在、``subdir`` 是绝对路径、同一目录下同一 stem 存在
            多个图片格式，或扫描到的文件不在 ``spec.root`` 之下。
        DatasetSpecError: 多任务数据集未指定 ``task``（由 ``spec.class_names`` 抛出）。
    """
    root_path = os.path.abspath(os.fspath(root))
    class_names = _class_names(spec, task)
    table = ClassTable(class_names)
    index_of = {name: position for position, name in enumerate(class_names)}

    base = root_path
    if task is not None:
        base = os.path.join(base, task.path)
    if subdir:
        if os.path.isabs(subdir):
            raise AnnotationError(
                f"subdir 必须是相对路径（当前为 {subdir!r}）",
                hint="subdir 相对 <root>/<task.path>，例如 train_data",
            )
        base = os.path.normpath(os.path.join(base, subdir))
    if not os.path.isdir(base):
        raise AnnotationError(
            f"类别根目录不存在或不是目录: {base}",
            hint="folder_per_class 布局要求 <root>[/<task.path>][/<subdir>]/<class>/*.jpg；"
            "请检查 spec.root / task.path / subdir 是否正确",
        )

    extensions = {extension.lower() for extension in spec.parse.image_ext}
    task_name = task.name if task is not None else None
    refs: list[SampleRef] = []
    for dirpath in iter_dirs(base, bool(spec.parse.recursive)):
        class_position = _class_position(dirpath, base, index_of)
        if class_position is None:
            # 不在任何类别目录下（例如 <base> 本身或杂项目录）：整目录没有样本
            continue
        names = list_image_names(dirpath, extensions)
        check_stem_conflicts(dirpath, names)
        for name in names:
            image_path = os.path.abspath(os.path.join(dirpath, name))
            refs.append(
                SampleRef(
                    image_path=image_path,
                    rel_path=rel_path_posix(image_path, spec.root),
                    task=task_name,
                    label=class_position,
                    meta={"class_name": table.name(class_position)},
                )
            )
    refs.sort(key=lambda ref: ref.rel_path)
    return refs


# ---------------------------------------------------------------------------
# 与 labelme 扫描器共用的目录遍历工具
# ---------------------------------------------------------------------------


def iter_dirs(base: str, recursive: bool) -> Iterator[str]:
    """遍历 ``base`` 下的目录（跳过 ``.`` 开头的隐藏目录），顺序确定。

    ``recursive=True`` 时深度优先遍历整个子树（``base`` 自身也会产出）；
    ``recursive=False`` 时只产出 ``base`` 自身与它的直接子目录。
    """
    if recursive:
        for dirpath, dirnames, _filenames in os.walk(base):
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            yield dirpath
        return
    yield base
    try:
        entries = sorted(os.scandir(base), key=lambda entry: entry.name)
    except OSError as exc:
        raise AnnotationError(
            f"无法读取目录: {base}（{exc}）",
            hint="请检查目录权限，或确认它是一个目录而不是文件",
        ) from exc
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        yield entry.path


def list_image_names(dirpath: str, extensions: set[str]) -> list[str]:
    """列出 ``dirpath`` 下扩展名命中的图片文件名（跳过隐藏文件，按名字升序）。"""
    try:
        names = sorted(os.listdir(dirpath))
    except OSError as exc:
        raise AnnotationError(
            f"无法读取目录: {dirpath}（{exc}）",
            hint="请检查目录权限",
        ) from exc
    images: list[str] = []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(dirpath, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in extensions:
            continue
        images.append(name)
    return images


def check_stem_conflicts(dirpath: str, names: list[str]) -> None:
    """拒绝"同一目录下同一 stem 的多个图片格式并存"（契约 P1-4）。

    Args:
        dirpath: 这些文件所在目录。
        names: 同一目录下扩展名命中的图片文件名。

    Raises:
        AnnotationError: 存在 ``x.jpg`` 与 ``x.png`` 之类无法判定归属的冲突。
    """
    by_stem: dict[str, list[str]] = {}
    for name in names:
        by_stem.setdefault(os.path.splitext(name)[0], []).append(name)
    for stem, group in by_stem.items():
        if len(group) < 2:
            continue
        paths = [os.path.join(dirpath, name) for name in sorted(group)]
        raise AnnotationError(
            f"同一 stem 的图片格式冲突: {stem!r} -> {paths}",
            hint="同一张图只应保留一种格式（.jpg 或 .png），否则会被当成两个样本，"
            "请删除多余文件后重新扫描",
        )


def rel_path_posix(path: str, spec_root: str) -> str:
    """返回相对 ``spec_root`` 的 posix 风格路径（契约 P1-0 的路径约定）。

    Raises:
        AnnotationError: ``path`` 不在 ``spec_root`` 之下（``rel_path`` 会失去意义）。
    """
    root = os.path.abspath(os.fspath(spec_root)) if spec_root else ""
    base = root or os.path.abspath(os.fspath(path))
    relative = os.path.relpath(os.path.abspath(os.fspath(path)), base)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise AnnotationError(
            f"扫描到的图片不在 spec.root 之下: {path}（spec.root={root}）",
            hint="请把 root 指向 spec.root 或其子目录；rel_path 必须相对 spec.root，"
            "切分与清单匹配都依赖这一点",
        )
    return relative.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _class_names(spec: DVXDatasetSpec, task: TaskSpec | None) -> list[str]:
    """取本次扫描使用的类别表（任务独立类别表优先）；空表直接报错，避免静默扫出 0 个样本。"""
    names = list(task.classes) if task is not None else spec.class_names()
    if not names:
        where = f"task {task.name!r} 的 classes" if task is not None else "spec 的 classes"
        raise DatasetSpecError(
            f"{where} 为空，无法按目录名匹配类别",
            hint="请在 dvx_dataset.yaml 中声明非空的 classes（顺序即 label id）",
        )
    return names


def _class_position(dirpath: str, base: str, index_of: Mapping[str, int]) -> int | None:
    """返回 ``dirpath`` 对应的类别 id（最近的类别祖先目录）；不在类别目录下返回 None。"""
    relative = os.path.relpath(dirpath, base)
    if relative == os.curdir:
        # <base> 自身不是类别目录：图片必须位于某个类别目录之内
        return None
    parts = relative.split(os.sep)
    for name in reversed(parts):
        position = index_of.get(name)
        if position is not None:
            return position
    return None
