# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集体检工具（契约 P1-9）。
"""DVX 数据集体检工具（契约 P1-9）。

对一份 ``dvx_dataset.yaml`` 执行"扫描 + 统计 + 可选切分预览"，只读，不移动任何文件：

- 基本信息：data_type / layout / root / 类别数 / 样本总数 / 每类样本数（检测、分割为
  每类目标数）。
- 标注质量：
  - detection：框总数、每样本框数分布、空 shapes、越界框、退化框、跳过形状；
  - segmentation：多边形数、按多边形面积估算的像素覆盖率、越界顶点、空 polygons、
    跳过形状；
  - classification：目录型数据无独立标注，只报告跳过形状为 0。
- ``--split-preview``：按 ``spec.splits`` 调用 ``build_splits``，打印每个 split 的样本数、
  每类分布与 warning。
- ``--json``：把同一份结果输出为 JSON（stdout 只有 JSON）。

严重问题（spec/标注扫描失败、未知类别、类别表为空、切分为空 train 等）会以退出码 1 结束；
缺 JSON 的图片会被跳过并告警、warning 均不影响退出码。

用法::

    python tools/check_dataset.py --spec project/data/det/dvx_dataset.yaml
    python tools/check_dataset.py --spec project/data/cls/dvx_dataset.yaml --split-preview
    python tools/check_dataset.py --spec project/data/seg/dvx_dataset.yaml --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

# 让 `python tools/check_dataset.py` 在仓库根目录之外也能 import 到本仓库的 dvisionix
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dvisionix.data.spec.scan import SampleRef, scan_samples  # noqa: E402
from dvisionix.data.spec.schema import DVXDatasetSpec, load_spec  # noqa: E402
from dvisionix.data.spec.split import build_splits  # noqa: E402
from dvisionix.errors import (  # noqa: E402
    AnnotationError,
    DatasetSpecError,
    DVisionixError,
    SplitError,
)

#: 浮点统计值的保留位数（只影响可读性，不影响判定）
ROUND_DIGITS = 6


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="检查 DVX 数据集规格、标注质量与切分结果（只读）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True, help="dvx_dataset.yaml 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（stdout 只有 JSON）")
    parser.add_argument(
        "--split-preview",
        action="store_true",
        help="按 spec.splits 构建切分并打印每个 split 的样本数与每类分布",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回进程退出码（0 通过 / 1 存在严重问题）。"""
    args = build_parser().parse_args(argv)

    try:
        spec = load_spec(args.spec)
    except DatasetSpecError as exc:
        return _report_fatal(args, os.path.abspath(args.spec), exc)

    try:
        refs = scan_samples(spec)
    except (DatasetSpecError, AnnotationError) as exc:
        return _report_fatal(args, spec.spec_path, exc, spec=spec)

    try:
        report = _build_report(spec, refs, split_preview=args.split_preview)
    except DVisionixError as exc:
        return _report_fatal(args, spec.spec_path, exc, spec=spec)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


def _report_fatal(
    args: argparse.Namespace,
    spec_path: str,
    exc: DVisionixError,
    spec: DVXDatasetSpec | None = None,
) -> int:
    """输出致命错误（--json 时仍保证 stdout 是合法 JSON），返回 1。"""
    error = str(exc)
    payload: dict[str, Any] = {
        "ok": False,
        "spec_path": spec_path,
        "errors": [error],
    }
    if spec is not None:
        payload.update(
            {
                "root": spec.root,
                "data_type": spec.data_type,
                "layout": spec.layout,
            }
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[check_dataset] 检查失败：\n{error}", file=sys.stderr)
        print("结论: 失败（退出码 1）", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------


def _build_report(spec: DVXDatasetSpec, refs: list[SampleRef], split_preview: bool) -> dict:
    """组装完整报告（可 JSON 序列化）。"""
    errors: list[str] = []
    warnings: list[str] = []
    if not refs:
        errors.append("未扫描到任何样本")

    classes_report = _classes_report(spec, refs)
    warnings.extend(classes_report.pop("_warnings", []))

    quality = _quality_report(spec, refs)
    warnings.extend(quality.pop("_warnings", []))

    split_report = None
    if split_preview:
        split_report = _split_report(spec, refs)
        warnings.extend(split_report["warnings"])
        if not split_report["ok"]:
            errors.append(f"切分失败: {split_report['error']}")
        elif split_report["configured"]:
            train_count = split_report["splits"].get("train", {}).get("count", 0)
            if train_count == 0:
                errors.append("切分产生的 train 为空")

    return {
        "ok": not errors,
        "spec_path": spec.spec_path,
        "root": spec.root,
        "data_type": spec.data_type,
        "layout": spec.layout,
        "seg_mode": spec.seg_mode,
        "ignore_index": spec.ignore_index,
        "num_classes": classes_report["num_classes"],
        "classes": classes_report,
        "total_samples": len(refs),
        "quality": quality,
        "split_preview": split_report,
        "errors": errors,
        "warnings": _dedupe(warnings),
    }


def _classes_report(spec: DVXDatasetSpec, refs: list[SampleRef]) -> dict:
    """统计类别表、每类样本数/目标数与缺失类别。"""
    warnings: list[str] = []
    if spec.data_type == "multi_task_classification":
        task_reports = []
        total_classes = 0
        for task in spec.tasks or []:
            counts = {name: 0 for name in task.classes}
            for ref in refs:
                if ref.task != task.name or ref.label is None:
                    continue
                name = _label_name(task.classes, ref.label)
                counts[name] = counts.get(name, 0) + 1
            missing = [name for name, count in counts.items() if count == 0]
            warnings.extend(f"任务 {task.name} 的类别 {name} 没有样本" for name in missing)
            task_reports.append(
                {
                    "name": task.name,
                    "path": task.path,
                    "num_classes": len(task.classes),
                    "per_class_samples": counts,
                    "per_class_objects": {},
                    "missing": missing,
                }
            )
            total_classes += len(task.classes)
        return {
            "num_classes": total_classes,
            "per_class_samples": {},
            "per_class_objects": {},
            "missing": [
                f"{item['name']}:{name}" for item in task_reports for name in item["missing"]
            ],
            "tasks": task_reports,
            "_warnings": warnings,
        }

    names = spec.class_names()
    if spec.data_type == "classification":
        counts = {name: 0 for name in names}
        unlabeled = 0
        for ref in refs:
            if ref.label is None:
                unlabeled += 1
                continue
            name = _label_name(names, ref.label)
            counts[name] = counts.get(name, 0) + 1
        if unlabeled:
            warnings.append(f"{unlabeled} 个分类样本没有 label")
        missing = [name for name, count in counts.items() if count == 0]
        warnings.extend(f"类别 {name} 没有样本" for name in missing)
        return {
            "num_classes": len(names),
            "per_class_samples": counts,
            "per_class_objects": {},
            "missing": missing,
            "_warnings": warnings,
        }

    objects = {name: 0 for name in names}
    unknown = 0
    for ref in refs:
        labels = ref.labels if spec.data_type == "detection" else ref.segment_labels
        if labels is None:
            continue
        for raw in labels:
            index = int(raw)
            if 0 <= index < len(names):
                objects[names[index]] += 1
            else:
                unknown += 1
    if unknown:
        warnings.append(f"{unknown} 个标注的类别 id 超出类别表范围（0..{len(names) - 1}）")
    ignored = set(spec.parse.labelme_ignore_labels) if spec.data_type == "segmentation" else set()
    # __background__ 等忽略标签不参与标注目标统计，缺目标不算异常
    missing = [name for name, count in objects.items() if count == 0 and name not in ignored]
    warnings.extend(f"类别 {name} 没有目标" for name in missing)
    return {
        "num_classes": len(names),
        "per_class_samples": {},
        "per_class_objects": objects,
        "missing": missing,
        "_warnings": warnings,
    }


def _quality_report(spec: DVXDatasetSpec, refs: list[SampleRef]) -> dict:
    """按数据类型统计标注质量；不适用时返回空量指标。"""
    if spec.data_type == "detection":
        return _detection_quality(refs)
    if spec.data_type == "segmentation":
        return _segmentation_quality(refs)
    return {
        "kind": "folder_per_class",
        "applicable": False,
        "skipped_shapes": 0,
        "samples_with_skipped_shapes": 0,
        "_warnings": [],
    }


def _detection_quality(refs: list[SampleRef]) -> dict:
    """统计检测框数量分布、越界/退化框、空 shapes 与跳过形状。"""
    box_counts: list[int] = []
    out_of_bounds = 0
    samples_with_oob = 0
    degenerate = 0
    missing_size = 0
    image_data = 0
    for ref in refs:
        boxes = ref.boxes if ref.boxes is not None else []
        count = len(boxes)
        box_counts.append(count)
        height, width = _image_size(ref)
        if height is None or width is None:
            missing_size += 1
        sample_oob = 0
        for box in boxes:
            x1, y1, x2, y2 = (float(box[index]) for index in range(4))
            if x2 <= x1 or y2 <= y1:
                degenerate += 1
            if height is not None and width is not None:
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    out_of_bounds += 1
                    sample_oob += 1
        if sample_oob:
            samples_with_oob += 1
        if ref.meta.get("image_data_present"):
            image_data += 1

    skipped, samples_with_skipped = _skipped_stats(refs)
    warnings: list[str] = []
    if missing_size:
        warnings.append(f"{missing_size} 个样本缺少图像尺寸，无法判断越界框")
    if degenerate:
        warnings.append(f"发现 {degenerate} 个退化框（x2<=x1 或 y2<=y1）")
    if out_of_bounds:
        warnings.append(f"发现 {out_of_bounds} 个越界框（共 {samples_with_oob} 个样本）")

    histogram = {}
    for count in sorted(set(box_counts)):
        histogram[str(count)] = box_counts.count(count)
    return {
        "kind": "detection",
        "applicable": True,
        "total_boxes": sum(box_counts),
        "samples_without_boxes": sum(1 for count in box_counts if count == 0),
        "box_count": _distribution(box_counts, as_int=True),
        "box_count_histogram": histogram,
        "out_of_bounds_boxes": out_of_bounds,
        "samples_with_out_of_bounds": samples_with_oob,
        "degenerate_boxes": degenerate,
        "missing_image_size_samples": missing_size,
        "skipped_shapes": skipped,
        "samples_with_skipped_shapes": samples_with_skipped,
        "samples_with_image_data": image_data,
        "_warnings": warnings,
    }


def _segmentation_quality(refs: list[SampleRef]) -> dict:
    """统计多边形数量、面积覆盖率估算、越界顶点、空 polygons 与跳过形状。"""
    polygon_counts: list[int] = []
    coverage: list[float] = []
    out_of_bounds_vertices = 0
    samples_with_oob = 0
    missing_size = 0
    image_data = 0
    for ref in refs:
        polygons = ref.segments or []
        polygon_counts.append(len(polygons))
        height, width = _image_size(ref)
        if height is None or width is None:
            missing_size += 1
        area = sum(_polygon_area(polygon) for polygon in polygons)
        if height is not None and width is not None and height > 0 and width > 0:
            coverage.append(area / float(height * width))
        sample_oob = 0
        for polygon in polygons:
            for point in polygon:
                x, y = float(point[0]), float(point[1])
                if height is not None and width is not None:
                    if x < 0 or y < 0 or x > width or y > height:
                        out_of_bounds_vertices += 1
                        sample_oob += 1
        if sample_oob:
            samples_with_oob += 1
        if ref.meta.get("image_data_present"):
            image_data += 1

    skipped, samples_with_skipped = _skipped_stats(refs)
    warnings: list[str] = []
    if missing_size:
        warnings.append(f"{missing_size} 个样本缺少图像尺寸，无法估算像素覆盖率")
    if out_of_bounds_vertices:
        warnings.append(
            f"发现 {out_of_bounds_vertices} 个越界多边形顶点（共 {samples_with_oob} 个样本）"
        )
    over_one = sum(1 for value in coverage if value > 1.0)
    if over_one:
        warnings.append(
            f"{over_one} 个样本的多边形面积和超过图像面积（覆盖率为估算值，重叠部分重复计入）"
        )

    return {
        "kind": "segmentation",
        "applicable": True,
        "total_polygons": sum(polygon_counts),
        "samples_without_polygons": sum(1 for count in polygon_counts if count == 0),
        "polygon_count": _distribution(polygon_counts, as_int=True),
        "pixel_coverage": _distribution(coverage),
        "samples_coverage_gt_one": over_one,
        "out_of_bounds_vertices": out_of_bounds_vertices,
        "samples_with_out_of_bounds": samples_with_oob,
        "missing_image_size_samples": missing_size,
        "skipped_shapes": skipped,
        "samples_with_skipped_shapes": samples_with_skipped,
        "samples_with_image_data": image_data,
        "_warnings": warnings,
    }


def _split_report(spec: DVXDatasetSpec, refs: list[SampleRef]) -> dict:
    """按 spec.splits 构建切分并提取每个 split 的统计与 warning。"""
    if spec.splits is None:
        return {
            "configured": False,
            "ok": True,
            "mode": None,
            "seed": None,
            "stratify": None,
            "total": len(refs),
            "splits": {},
            "warnings": ["spec 未配置 splits，已跳过切分预览"],
            "error": None,
        }
    try:
        result = build_splits(refs, spec.splits, spec)
    except SplitError as exc:
        return {
            "configured": True,
            "ok": False,
            "mode": spec.splits.mode,
            "seed": spec.splits.seed,
            "stratify": spec.splits.stratify,
            "total": len(refs),
            "splits": {},
            "warnings": [],
            "error": str(exc),
        }

    stats = result.stats if isinstance(result.stats, Mapping) else {}
    split_stats = stats.get("splits") if isinstance(stats.get("splits"), Mapping) else {}
    splits: dict[str, dict[str, Any]] = {}
    for name, indices in result.splits.items():
        info = split_stats.get(name) if isinstance(split_stats, Mapping) else None
        info = info if isinstance(info, Mapping) else {}
        per_class = info.get("per_class")
        splits[name] = {
            "count": int(info.get("count", len(indices))),
            "per_class": dict(per_class) if isinstance(per_class, Mapping) else {},
        }
    return {
        "configured": True,
        "ok": True,
        "mode": result.mode,
        "seed": result.seed,
        "stratify": spec.splits.stratify,
        "total": len(refs),
        "splits": splits,
        "warnings": [str(item) for item in (stats.get("warnings") or [])],
        "error": None,
    }


# ---------------------------------------------------------------------------
# 人类可读输出
# ---------------------------------------------------------------------------


def _print_report(report: dict) -> None:
    """把报告打印为多段中文文本。"""
    print("DVX 数据集检查")
    print(f"  规格文件: {report['spec_path']}")
    print(f"  数据类型: {report['data_type']}    布局: {report['layout']}")
    if report["data_type"] == "segmentation":
        print(f"  分割模式: {report['seg_mode']}    ignore_index: {report['ignore_index']}")
    print(f"  根目录: {report['root']}")
    print(f"  类别数: {report['num_classes']}")
    print(f"  样本总数: {report['total_samples']}")

    classes = report["classes"]
    if classes.get("tasks"):
        for task in classes["tasks"]:
            print(f"  任务 {task['name']}（{task['path']}）：{task['num_classes']} 类")
            for name, count in task["per_class_samples"].items():
                print(f"    {name}: {count}")
    elif report["data_type"] == "classification":
        print("  每类样本数:")
        for name, count in classes["per_class_samples"].items():
            ratio = count / report["total_samples"] if report["total_samples"] else 0.0
            print(f"    {name}: {count} ({ratio:.1%})")
    else:
        print("  每类目标数:")
        for name, count in classes["per_class_objects"].items():
            print(f"    {name}: {count}")

    _print_quality(report["quality"], report["data_type"])
    if report["split_preview"] is not None:
        _print_split_preview(report["split_preview"])

    warnings = report["warnings"]
    if warnings:
        print(f"警告（{len(warnings)}）:")
        for item in warnings:
            print(f"  - {item}")
    if report["errors"]:
        print(f"错误（{len(report['errors'])}）:")
        for item in report["errors"]:
            print(f"  - {item}")
    print("结论: " + ("通过" if report["ok"] else "失败（存在严重问题）"))


def _print_quality(quality: dict, data_type: str) -> None:
    """打印标注质量段落。"""
    if data_type not in ("detection", "segmentation"):
        print("标注质量: 无（folder_per_class 目录型分类没有独立标注文件）")
        return
    print("标注质量:")
    if data_type == "detection":
        dist = quality["box_count"]
        print(f"  检测框总数: {quality['total_boxes']}")
        print(
            "  每样本框数: "
            f"min={dist['min']} median={dist['median']} mean={dist['mean']} max={dist['max']}"
        )
        histogram = ", ".join(
            f"{key}={value}" for key, value in quality["box_count_histogram"].items()
        )
        print(f"  框数分布: {histogram}")
        print(f"  空 shapes 样本: {quality['samples_without_boxes']}")
        print(
            f"  越界框: {quality['out_of_bounds_boxes']}"
            f"（涉及 {quality['samples_with_out_of_bounds']} 个样本）"
        )
        print(f"  退化框（x2<=x1 或 y2<=y1）: {quality['degenerate_boxes']}")
        print(f"  缺失图像尺寸样本: {quality['missing_image_size_samples']}")
        print(f"  内嵌 imageData 样本: {quality['samples_with_image_data']}")
    else:
        dist = quality["polygon_count"]
        coverage = quality["pixel_coverage"]
        print(f"  多边形总数: {quality['total_polygons']}")
        print(
            "  每样本多边形数: "
            f"min={dist['min']} median={dist['median']} mean={dist['mean']} max={dist['max']}"
        )
        print(
            "  像素覆盖率（按多边形面积估算）: "
            f"mean={coverage['mean']} min={coverage['min']} max={coverage['max']}"
        )
        print(f"  覆盖率 > 1 的样本: {quality['samples_coverage_gt_one']}")
        print(f"  空 polygons 样本: {quality['samples_without_polygons']}")
        print(
            f"  越界顶点: {quality['out_of_bounds_vertices']}"
            f"（涉及 {quality['samples_with_out_of_bounds']} 个样本）"
        )
        print(f"  缺失图像尺寸样本: {quality['missing_image_size_samples']}")
        print(f"  内嵌 imageData 样本: {quality['samples_with_image_data']}")
    print(
        f"  跳过的形状: {quality['skipped_shapes']}"
        f"（涉及 {quality['samples_with_skipped_shapes']} 个样本）"
    )


def _print_split_preview(split: dict) -> None:
    """打印切分预览段落。"""
    if not split["configured"]:
        print("切分预览: 未配置（spec.splits 为空）")
        return
    if not split["ok"]:
        print("切分预览: 失败")
        print(f"  {split['error']}")
        return
    print("切分预览:")
    detail = f"  模式: {split['mode']}"
    if split["seed"] is not None:
        detail += f"（seed={split['seed']}"
        detail += f"，stratify={split['stratify']}）" if split["stratify"] else "）"
    print(detail)
    for name, info in split["splits"].items():
        print(f"  {name}: {info['count']} 个样本")
        if info["per_class"]:
            per_class = ", ".join(f"{key}={value}" for key, value in info["per_class"].items())
            print(f"    每类分布: {per_class}")
    warnings = split["warnings"]
    if warnings:
        print(f"  切分警告（{len(warnings)}）:")
        for item in warnings:
            print(f"    - {item}")
    else:
        print("  切分警告: 无")


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _label_name(names: Sequence[str], index: int) -> str:
    """把类别 id 映射为名字；越界时返回可读占位符（扫描器正常不会产生越界 id）。"""
    if 0 <= index < len(names):
        return names[index]
    return f"<label:{index}>"


def _image_size(ref: SampleRef) -> tuple[int | None, int | None]:
    """返回 ``(H, W)``；缺失时用 ``(None, None)``。"""
    if ref.image_size is None:
        return None, None
    try:
        height, width = ref.image_size
    except (TypeError, ValueError):
        return None, None
    return int(height), int(width)


def _polygon_area(polygon: Any) -> float:
    """用鞋带公式计算多边形面积（像素²，取绝对值；少于 3 点记 0）。"""
    points = list(polygon)
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        x1, y1 = float(point[0]), float(point[1])
        nxt = points[(index + 1) % len(points)]
        x2, y2 = float(nxt[0]), float(nxt[1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _distribution(values: Iterable[float], *, as_int: bool = False) -> dict[str, float | int]:
    """返回 min / median / mean / max / count（空序列返回全 0）。

    Args:
        values: 待统计的数值序列。
        as_int: 为 True 时把整数结果还原为 ``int``（框数/多边形数等计数更易读）。
    """

    def _cast(value: float) -> float | int:
        rounded = _round(value)
        if as_int and float(rounded).is_integer():
            return int(rounded)
        return rounded

    items = [float(value) for value in values]
    if not items:
        return {"min": 0, "median": 0, "mean": 0, "max": 0, "count": 0}
    return {
        "min": _cast(min(items)),
        "median": _cast(statistics.median(items)),
        "mean": _cast(sum(items) / len(items)),
        "max": _cast(max(items)),
        "count": len(items),
    }


def _round(value: float) -> float:
    """把浮点统计值四舍五入到固定小数位。"""
    return round(float(value), ROUND_DIGITS)


def _skipped_stats(refs: Sequence[SampleRef]) -> tuple[int, int]:
    """返回 ``(跳过的形状总数, 涉及样本数)``。"""
    total = 0
    samples = 0
    for ref in refs:
        skipped = int(ref.meta.get("skipped_shapes", 0) or 0)
        if skipped:
            total += skipped
            samples += 1
    return total, samples


def _dedupe(items: Sequence[str]) -> list[str]:
    """按首次出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
