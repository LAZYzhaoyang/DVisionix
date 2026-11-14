# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: DVX 数据集解析器（folder / labelme / rasterize / scan）单元测试。
"""DVX 数据集解析器单元测试（契约 P1-6 / P1-11）。

覆盖范围：

- ``folder``：类别目录识别、非类目录忽略、最近类祖先、递归开关、多任务 task.path、
  ``subdir``、中文类名、扩展名大小写、隐藏目录、同 stem 冲突、越界 root；
- ``labelme``：rectangle（2 点/旋转 4 点）/ polygon、形状类型过滤、ignore_labels、
  未知标签、缺 JSON 跳过并告警、JSON 语法错误、扩展名冲突、imageData（ignore/error）、
  image_size 与 cv2 兜底、空 shapes、退化多边形、非法矩形；
- ``rasterize``：覆盖顺序、越界裁剪、空列表、浮点取整、dtype/shape、参数校验；
- ``scan``：按 layout 分派、多任务聚合、显式 subdir（``dir_map`` 不再推断子目录）、
  重复 rel_path、``SampleRef`` 默认值。

全部夹具用 ``tmp_path`` + numpy + cv2 现场生成，**不依赖** ``project/data``。
"""

import json

import cv2
import numpy as np
import pytest

from dvisionix.data.parsers.folder import scan_folder_per_class
from dvisionix.data.parsers.labelme import parse_labelme_file, scan_labelme
from dvisionix.data.parsers.rasterize import polygons_to_mask
from dvisionix.data.spec.classes import ClassTable
from dvisionix.data.spec.scan import SampleRef, scan_samples
from dvisionix.data.spec.schema import DVXDatasetSpec, ParseConfig, SplitConfig, TaskSpec
from dvisionix.data.spec.split import build_splits
from dvisionix.errors import AnnotationError, DatasetSpecError, SplitError

pytestmark = pytest.mark.unit

IGNORE_LABELS = ("__ignore__", "__background__")


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _write_image(path, height=8, width=8):
    """写一张真实可解码的小图（用 imencode 避免非 ASCII 路径写入失败）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(path.suffix.lower(), image)
    assert ok, f"cv2.imencode 失败: {path}"
    path.write_bytes(buffer.tobytes())
    return path


def _shape(label, shape_type, points):
    """构造一个 labelme shape 字典。"""
    return {
        "label": label,
        "shape_type": shape_type,
        "points": points,
        "group_id": None,
        "flags": {},
    }


def _rect(label, x1, y1, x2, y2):
    """构造两点矩形（labelme 默认导出形式）。"""
    return _shape(label, "rectangle", [[x1, y1], [x2, y2]])


def _polygon(label, points):
    """构造多边形 shape。"""
    return _shape(label, "polygon", points)


def _write_labelme(path, shapes, image_name="x.jpg", height=8, width=8, image_data=None):
    """写一个手写 labelme JSON（``height``/``width`` 传 None 表示不写该字段）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": image_data,
    }
    if height is not None:
        document["imageHeight"] = height
    if width is not None:
        document["imageWidth"] = width
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _spec(
    tmp_path,
    data_type="classification",
    classes=("cat", "dog"),
    tasks=None,
    splits=None,
    parse=None,
    root=None,
):
    """构造一个指向 ``tmp_path/data`` 的 spec（目录不存在时现场创建）。"""
    root_path = root or (tmp_path / "data")
    root_path.mkdir(parents=True, exist_ok=True)
    layout = (
        "folder_per_class"
        if data_type in ("classification", "multi_task_classification")
        else "labelme"
    )
    task_specs = None
    if tasks is not None:
        task_specs = [
            TaskSpec(name=name, path=path, classes=list(task_classes))
            for name, path, task_classes in tasks
        ]
    return DVXDatasetSpec(
        version=1,
        data_type=data_type,
        layout=layout,
        root=str(root_path),
        classes=None if data_type == "multi_task_classification" else list(classes),
        tasks=task_specs,
        splits=splits,
        parse=parse or ParseConfig(),
        spec_path=str(tmp_path / "dvx_dataset.yaml"),
    )


def _image_refs(refs):
    """只取 rel_path 列表，便于断言顺序。"""
    return [ref.rel_path for ref in refs]


# ---------------------------------------------------------------------------
# rasterize
# ---------------------------------------------------------------------------


def test_polygons_to_mask_fills_label_and_dtype():
    polygon = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float32)
    mask = polygons_to_mask([polygon], [3], 8, 8)
    assert mask.shape == (8, 8)
    assert mask.dtype == np.int64
    assert mask[3, 3] == 3
    assert mask[0, 0] == 0 and mask[7, 7] == 0


def test_polygons_to_mask_later_polygon_overwrites_earlier():
    background_polygon = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float32)
    inner_polygon = np.array([[2, 2], [5, 2], [5, 5], [2, 5]], dtype=np.float32)
    mask = polygons_to_mask([background_polygon, inner_polygon], [1, 2], 8, 8)
    assert mask[3, 3] == 2  # 后写覆盖先写
    assert mask[0, 0] == 1
    # 顺序反过来时结论相反，证明 "后写覆盖" 而非 "大者胜"
    reversed_mask = polygons_to_mask([inner_polygon, background_polygon], [2, 1], 8, 8)
    assert reversed_mask[3, 3] == 1


def test_polygons_to_mask_clips_out_of_bounds_coordinates():
    polygon = np.array([[-4.0, -4.0], [20.0, -4.0], [20.0, 20.0], [-4.0, 20.0]], dtype=np.float32)
    mask = polygons_to_mask([polygon], [5], 5, 6)
    assert mask.shape == (5, 6)
    assert np.all(mask == 5)  # 越界部分被裁剪到图像范围，整图被填充


def test_polygons_to_mask_clamps_extreme_coordinates():
    # 极值坐标先在浮点域裁剪，避免 astype(int32) 溢出成反号像素
    polygon = np.array([[-1e9, -1e9], [1e9, -1e9], [1e9, 1e9], [-1e9, 1e9]], dtype=np.float32)
    mask = polygons_to_mask([polygon], [4], 3, 5)
    assert np.all(mask == 4)


def test_polygons_to_mask_empty_returns_full_background():
    mask = polygons_to_mask([], [], 4, 5, background=255)
    assert mask.shape == (4, 5)
    assert mask.dtype == np.int64
    assert np.all(mask == 255)
    # 空 numpy 标签数组同样可用（segments=[] + segment_labels=空数组）
    empty_labels = np.zeros((0,), dtype=np.int64)
    assert np.all(polygons_to_mask([], empty_labels, 4, 5) == 0)
    # 空多边形本身被跳过，不影响背景与 dtype
    assert np.all(polygons_to_mask([np.zeros((0, 2))], [1], 4, 5, background=7) == 7)


def test_polygons_to_mask_rounds_float_coordinates():
    polygon = np.array([[1.6, 1.6], [4.4, 1.6], [4.4, 4.4], [1.6, 4.4]], dtype=np.float32)
    mask = polygons_to_mask([polygon], [1], 8, 8)
    assert mask[2, 2] == 1  # rint(1.6) == 2
    assert mask[1, 1] == 0
    assert mask[6, 6] == 0


def test_polygons_to_mask_rejects_label_length_mismatch():
    polygon = np.array([[1, 1], [4, 1], [4, 4]], dtype=np.float32)
    with pytest.raises(AnnotationError, match="数量"):
        polygons_to_mask([polygon], [1, 2], 8, 8)


def test_polygons_to_mask_rejects_bad_polygon_shape():
    with pytest.raises(AnnotationError, match=r"\(P, 2\)"):
        polygons_to_mask([np.array([1, 2, 3], dtype=np.float32)], [1], 8, 8)


def test_polygons_to_mask_rejects_nan_coordinates():
    polygon = np.array([[1.0, np.nan], [4.0, 1.0], [4.0, 4.0]], dtype=np.float32)
    with pytest.raises(AnnotationError, match="NaN"):
        polygons_to_mask([polygon], [1], 8, 8)


def test_polygons_to_mask_rejects_invalid_size():
    with pytest.raises(AnnotationError, match="height"):
        polygons_to_mask([], [], 0, 8)
    with pytest.raises(AnnotationError, match="width"):
        polygons_to_mask([], [], 8, 8.0)


# ---------------------------------------------------------------------------
# folder_per_class
# ---------------------------------------------------------------------------


def test_scan_folder_basic_fields_and_order(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "b.jpg")
    _write_image(root / "cat" / "a.jpg")
    _write_image(root / "dog" / "c.jpg")
    spec = _spec(tmp_path, classes=["cat", "dog"], root=root)

    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/a.jpg", "cat/b.jpg", "dog/c.jpg"]
    assert _image_refs(refs) == sorted(_image_refs(refs))
    assert [ref.label for ref in refs] == [0, 0, 1]
    first = refs[0]
    assert first.image_path == str((root / "cat" / "a.jpg").resolve())
    assert first.task is None
    assert first.ann_path is None
    assert first.image_size is None
    assert first.boxes is None and first.labels is None and first.segments is None
    assert first.meta["class_name"] == "cat"


def test_scan_folder_chinese_class_names(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "幻影灰" / "1.jpg")
    _write_image(root / "科技灰" / "2.jpg")
    spec = _spec(tmp_path, classes=["幻影灰", "科技灰"], root=root)

    refs = scan_folder_per_class(spec.root, spec)

    assert [ref.rel_path for ref in refs] == ["幻影灰/1.jpg", "科技灰/2.jpg"]
    assert [ref.label for ref in refs] == [0, 1]
    assert [ref.meta["class_name"] for ref in refs] == ["幻影灰", "科技灰"]


def test_scan_folder_ignores_non_class_directories(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "a.jpg")
    _write_image(root / "loose.jpg")  # 直接躺在 root 下：不在类别目录内
    _write_image(root / "junk" / "b.jpg")  # junk 不是类别：整目录没有样本
    _write_image(root / "train_data" / "c.jpg")  # train_data 本身不是类别
    _write_image(root / "train_data" / "train" / "cat" / "d.jpg")  # 最近的类祖先才是类别

    spec = _spec(tmp_path, classes=["cat", "dog"], root=root)
    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/a.jpg", "train_data/train/cat/d.jpg"]
    assert [ref.label for ref in refs] == [0, 0]


def test_scan_folder_uses_nearest_class_ancestor(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "sub" / "deep" / "x.jpg")
    spec = _spec(tmp_path, classes=["cat", "dog"], root=root)

    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/sub/deep/x.jpg"]
    assert refs[0].label == 0


def test_scan_folder_non_recursive_only_one_level(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "a.jpg")
    _write_image(root / "cat" / "sub" / "b.jpg")
    _write_image(root / "dog" / "c.jpg")
    _write_image(root / "train_data" / "train" / "cat" / "d.jpg")
    spec = _spec(tmp_path, classes=["cat", "dog"], root=root, parse=ParseConfig(recursive=False))

    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/a.jpg", "dog/c.jpg"]


def test_scan_folder_multitask_uses_task_path_and_classes(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "T_licence" / "无" / "a.jpg")
    _write_image(root / "T_licence" / "短牌照" / "b.jpg")
    _write_image(root / "T_logo" / "一体奇瑞标" / "c.jpg")
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[
            ("licence", "T_licence", ["无", "短牌照"]),
            ("logo", "T_logo", ["一体奇瑞标"]),
        ],
        root=root,
    )
    task = spec.tasks[0]

    licence_refs = scan_folder_per_class(spec.root, spec, task=task)
    logo_refs = scan_folder_per_class(spec.root, spec, task=spec.tasks[1])

    assert _image_refs(licence_refs) == ["T_licence/无/a.jpg", "T_licence/短牌照/b.jpg"]
    assert [ref.task for ref in licence_refs] == ["licence", "licence"]
    assert [ref.label for ref in licence_refs] == [0, 1]
    assert _image_refs(logo_refs) == ["T_logo/一体奇瑞标/c.jpg"]
    assert logo_refs[0].label == 0


def test_scan_folder_multitask_requires_task(tmp_path):
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[("licence", "T_licence", ["无"])],
    )
    with pytest.raises(DatasetSpecError, match="必须指定 task"):
        scan_folder_per_class(spec.root, spec)


def test_scan_folder_subdir_limits_scan(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "T" / "cat" / "flat.jpg")
    _write_image(root / "T" / "train_data" / "train" / "cat" / "train.jpg")
    _write_image(root / "T" / "train_data" / "val" / "cat" / "val.jpg")
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[("licence", "T", ["cat"])],
        root=root,
    )
    task = spec.tasks[0]

    limited = scan_folder_per_class(spec.root, spec, task=task, subdir="train_data")
    full = scan_folder_per_class(spec.root, spec, task=task)

    assert _image_refs(limited) == [
        "T/train_data/train/cat/train.jpg",
        "T/train_data/val/cat/val.jpg",
    ]
    assert len(full) == 3


def test_scan_folder_subdir_must_be_relative(tmp_path):
    spec = _spec(tmp_path)
    with pytest.raises(AnnotationError, match="相对路径"):
        scan_folder_per_class(spec.root, spec, subdir=str(tmp_path / "data"))


def test_scan_folder_missing_class_root(tmp_path):
    spec = _spec(tmp_path, tasks=[("licence", "T_missing", ["无"])])
    task = TaskSpec(name="licence", path="T_missing", classes=["无"])
    with pytest.raises(AnnotationError, match="不存在"):
        scan_folder_per_class(spec.root, spec, task=task)


def test_scan_folder_stem_conflict(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "x.jpg")
    _write_image(root / "cat" / "x.png")
    spec = _spec(tmp_path, classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="冲突") as excinfo:
        scan_folder_per_class(spec.root, spec)

    message = str(excinfo.value)
    assert "x.jpg" in message and "x.png" in message


def test_scan_folder_extension_match_is_case_insensitive(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "X.JPG")
    spec = _spec(tmp_path, classes=["cat"], root=root, parse=ParseConfig(image_ext=(".jpg",)))

    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/X.JPG"]


def test_scan_folder_skips_hidden_dirs_and_files(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "keep.jpg")
    _write_image(root / "cat" / ".hidden.jpg")
    _write_image(root / ".hidden_dir" / "cat" / "skip.jpg")
    spec = _spec(tmp_path, classes=["cat"], root=root)

    refs = scan_folder_per_class(spec.root, spec)

    assert _image_refs(refs) == ["cat/keep.jpg"]


def test_scan_folder_rejects_empty_class_table(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "a.jpg")
    spec = _spec(tmp_path, classes=["cat"], root=root)
    spec.classes = []

    with pytest.raises(DatasetSpecError, match="为空"):
        scan_folder_per_class(spec.root, spec)


def test_scan_folder_rejects_root_outside_spec_root(tmp_path):
    data_root = tmp_path / "data"
    other_root = tmp_path / "other"
    _write_image(other_root / "cat" / "a.jpg")
    spec = _spec(tmp_path, classes=["cat"], root=data_root)

    with pytest.raises(AnnotationError, match="不在 spec.root 之下"):
        scan_folder_per_class(str(other_root), spec)


# ---------------------------------------------------------------------------
# labelme
# ---------------------------------------------------------------------------


def _labelme_dataset(tmp_path, shapes, data_type="detection", classes=("cat", "dog"), **kwargs):
    """造一个"一张图 + 一个 json"的 labelme 数据集，返回 (spec, ClassTable)。"""
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(root / "A" / "x.json", shapes, image_name="x.jpg", **kwargs)
    spec = _spec(tmp_path, data_type=data_type, classes=classes, root=root)
    return spec, ClassTable(list(classes))


def test_scan_labelme_detection_rectangles(tmp_path):
    shapes = [
        _rect("cat", 1, 2, 5, 6),
        # 旋转矩形：4 个角点，取 min/max 作为 xyxy
        _shape("dog", "rectangle", [[6, 6], [2, 2], [6, 2], [2, 6]]),
    ]
    spec, table = _labelme_dataset(tmp_path, shapes, height=16, width=24)

    refs = scan_labelme(spec.root, spec, table)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.rel_path == "A/x.jpg"
    assert ref.image_path == str((tmp_path / "data" / "A" / "x.jpg").resolve())
    assert ref.ann_path == str((tmp_path / "data" / "A" / "x.json").resolve())
    assert ref.image_size == (16, 24)  # 来自 labelme 的 imageHeight/imageWidth（H, W）
    assert ref.boxes.dtype == np.float32 and ref.boxes.shape == (2, 4)
    np.testing.assert_allclose(ref.boxes, [[1, 2, 5, 6], [2, 2, 6, 6]])
    assert ref.labels.dtype == np.int64
    assert ref.labels.tolist() == [0, 1]
    assert ref.segments is None and ref.segment_labels is None
    assert ref.task is None and ref.label is None
    assert ref.meta["skipped_shapes"] == 0
    assert ref.meta["shape_count"] == 2
    assert ref.meta["image_data_present"] is False


def test_scan_labelme_segmentation_polygons(tmp_path):
    shapes = [
        _polygon("cat", [[1, 1], [6, 1], [6, 6]]),
        _polygon("dog", [[2, 2], [5, 2], [5, 5], [2, 5]]),
    ]
    spec, table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    refs = scan_labelme(spec.root, spec, table)

    ref = refs[0]
    assert ref.boxes is None and ref.labels is None
    assert len(ref.segments) == 2
    assert ref.segments[0].dtype == np.float32
    assert ref.segments[0].shape == (3, 2)
    assert ref.segments[1].shape == (4, 2)
    assert ref.segment_labels.dtype == np.int64
    assert ref.segment_labels.tolist() == [0, 1]
    assert ref.meta["shape_count"] == 2


def test_scan_labelme_detection_skips_polygon(tmp_path):
    shapes = [_rect("cat", 1, 1, 4, 4), _polygon("dog", [[1, 1], [5, 1], [5, 5]])]
    spec, table = _labelme_dataset(tmp_path, shapes)

    ref = scan_labelme(spec.root, spec, table)[0]

    assert ref.boxes.shape == (1, 4)
    assert ref.labels.tolist() == [0]
    assert ref.meta["skipped_shapes"] == 1


def test_scan_labelme_segmentation_skips_rectangle(tmp_path):
    shapes = [_polygon("dog", [[1, 1], [5, 1], [5, 5]]), _rect("cat", 1, 1, 4, 4)]
    spec, table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    ref = scan_labelme(spec.root, spec, table)[0]

    assert len(ref.segments) == 1
    assert ref.segment_labels.tolist() == [1]
    assert ref.meta["skipped_shapes"] == 1


def test_scan_labelme_skips_other_shape_types(tmp_path):
    shapes = [
        _shape("cat", "circle", [[2, 2], [4, 4], [0, 0]]),
        _shape("cat", "line", [[1, 1], [4, 4]]),
        _shape("cat", "point", [[2, 2]]),
        _rect("dog", 1, 1, 4, 4),
    ]
    spec, table = _labelme_dataset(tmp_path, shapes)

    ref = scan_labelme(spec.root, spec, table)[0]

    assert ref.boxes.shape == (1, 4)
    assert ref.meta["skipped_shapes"] == 3


def test_scan_labelme_degenerate_polygon_counted_as_skipped(tmp_path):
    shapes = [_polygon("cat", [[1, 1], [4, 4]]), _polygon("dog", [[1, 1], [5, 1], [5, 5]])]
    spec, table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    ref = scan_labelme(spec.root, spec, table)[0]

    assert len(ref.segments) == 1
    assert ref.meta["skipped_shapes"] == 1


def test_scan_labelme_missing_json_is_skipped_with_warning(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning, match="缺少同名 labelme JSON") as record:
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert refs == []
    assert len(record) == 1  # 只发一次汇总告警
    message = str(record[0].message)
    assert "1 张图片" in message and "已跳过" in message
    assert "A/x.jpg" in message


def test_scan_labelme_mixed_missing_json_returns_matched(tmp_path):
    """1 张有 JSON、1 张没有：只返回有标注的那条，并告警被跳过的一项。"""
    root = tmp_path / "data"
    _write_image(root / "A" / "has.jpg")
    _write_labelme(root / "A" / "has.json", [_rect("cat", 1, 1, 4, 4)])
    _write_image(root / "A" / "missing.jpg")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning, match="A/missing.jpg") as record:
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert _image_refs(refs) == ["A/has.jpg"]
    assert refs[0].boxes.shape == (1, 4)
    assert refs[0].labels.tolist() == [0]
    assert "1 张图片" in str(record[0].message)


def test_scan_labelme_all_missing_json_returns_empty_and_warns(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "a.jpg")
    _write_image(root / "A" / "b.jpg")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning, match="2 张图片缺少同名 labelme JSON") as record:
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert refs == []
    message = str(record[0].message)
    assert "A/a.jpg" in message and "A/b.jpg" in message


def test_scan_labelme_missing_json_samples_capped_at_ten(tmp_path):
    root = tmp_path / "data"
    for index in range(12):
        _write_image(root / "A" / f"img_{index:02d}.jpg")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning) as record:
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert refs == []
    message = str(record[0].message)
    assert "12 张图片" in message
    assert "A/img_00.jpg" in message and "A/img_09.jpg" in message
    assert "A/img_10.jpg" not in message  # 最多列 10 个示例
    assert "其余省略" in message


def test_scan_labelme_missing_json_keeps_empty_shapes_as_negative_sample(tmp_path):
    """空 shapes 的 JSON 是负样本，不因同目录其他图片缺 JSON 而被跳过。"""
    root = tmp_path / "data"
    _write_image(root / "A" / "empty.jpg")
    _write_labelme(root / "A" / "empty.json", [])
    _write_image(root / "A" / "nojson.jpg")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning, match="A/nojson.jpg"):
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert _image_refs(refs) == ["A/empty.jpg"]
    assert refs[0].boxes.shape == (0, 4)
    assert refs[0].meta["shape_count"] == 0


def test_scan_labelme_stem_conflict(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_image(root / "A" / "x.png")
    _write_labelme(root / "A" / "x.json", [_rect("cat", 1, 1, 4, 4)])
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="冲突"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_unknown_labels_listed(tmp_path):
    shapes = [_rect("fish", 1, 1, 4, 4), _rect("bird", 1, 1, 4, 4), _rect("cat", 1, 1, 4, 4)]
    spec, table = _labelme_dataset(tmp_path, shapes)

    with pytest.raises(AnnotationError, match="未登记类别") as excinfo:
        scan_labelme(spec.root, spec, table)

    message = str(excinfo.value)
    assert "fish" in message and "bird" in message
    assert "（2 个）" in message


def test_scan_labelme_unknown_labels_capped_at_five_samples(tmp_path):
    unknown = [f"unknown_{index:02d}" for index in range(7)]
    shapes = [_rect(label, 1, 1, 4, 4) for label in unknown]
    spec, table = _labelme_dataset(tmp_path, shapes)

    with pytest.raises(AnnotationError, match="未登记类别") as excinfo:
        scan_labelme(spec.root, spec, table)

    message = str(excinfo.value)
    assert "（7 个）" in message
    assert "unknown_04" in message
    assert "unknown_05" not in message  # 最多列 5 个样例


def test_scan_labelme_ignore_labels_dropped(tmp_path):
    shapes = [
        _polygon("__ignore__", [[1, 1], [5, 1], [5, 5]]),
        _rect("__background__", 1, 1, 4, 4),
        _polygon("cat", [[1, 1], [6, 1], [6, 6]]),
    ]
    spec, table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    ref = scan_labelme(spec.root, spec, table)[0]

    assert len(ref.segments) == 1
    assert ref.segment_labels.tolist() == [0]
    assert ref.meta["skipped_shapes"] == 0  # 忽略的标签不计入跳过数


def test_scan_labelme_image_data_ignored_and_counted(tmp_path):
    shapes = [_rect("cat", 1, 1, 4, 4)]
    spec, table = _labelme_dataset(tmp_path, shapes, image_data="QUJD")

    with pytest.warns(UserWarning, match="imageData"):
        refs = scan_labelme(spec.root, spec, table)

    assert refs[0].meta["image_data_present"] is True
    assert refs[0].boxes.shape == (1, 4)


def test_scan_labelme_image_data_error_mode(tmp_path):
    shapes = [_rect("cat", 1, 1, 4, 4)]
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(root / "A" / "x.json", shapes, image_name="x.jpg", image_data="QUJD")
    spec = _spec(
        tmp_path,
        data_type="detection",
        classes=["cat"],
        root=root,
        parse=ParseConfig(labelme_image_data="error"),
    )
    table = ClassTable(["cat"])

    with pytest.raises(AnnotationError, match="imageData"):
        scan_labelme(spec.root, spec, table)


def test_scan_labelme_invalid_json(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    (root / "A" / "x.json").write_text("{not valid json", encoding="utf-8")
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="解析失败") as excinfo:
        scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert "x.json" in str(excinfo.value)


def test_scan_labelme_empty_shapes_is_negative_sample(tmp_path):
    spec, table = _labelme_dataset(tmp_path, [])
    ref = scan_labelme(spec.root, spec, table)[0]
    assert ref.boxes.shape == (0, 4)
    assert ref.boxes.dtype == np.float32
    assert ref.labels.shape == (0,)
    assert ref.labels.dtype == np.int64
    assert ref.meta["shape_count"] == 0


def test_scan_labelme_empty_shapes_segmentation(tmp_path):
    spec, table = _labelme_dataset(tmp_path, [], data_type="segmentation")
    ref = scan_labelme(spec.root, spec, table)[0]
    assert ref.segments == []
    assert ref.segment_labels.shape == (0,)
    assert ref.segment_labels.dtype == np.int64


def test_scan_labelme_image_size_fallback_to_cv2(tmp_path):
    shapes = [_rect("cat", 1, 1, 4, 4)]
    spec, table = _labelme_dataset(tmp_path, shapes, height=None, width=None)

    with pytest.warns(UserWarning, match="imageWidth/imageHeight"):
        refs = scan_labelme(spec.root, spec, table)

    assert refs[0].image_size == (8, 8)  # 兜底用图片实际尺寸


def test_scan_labelme_malformed_rectangle_rejected(tmp_path):
    shapes = [_shape("cat", "rectangle", [[1, 1]])]
    spec, table = _labelme_dataset(tmp_path, shapes)

    with pytest.raises(AnnotationError, match="顶点不足"):
        scan_labelme(spec.root, spec, table)


def test_scan_labelme_malformed_points_rejected(tmp_path):
    shapes = [_shape("cat", "rectangle", [[1, 1, 2], [3, 4, 5]])]
    spec, table = _labelme_dataset(tmp_path, shapes)

    with pytest.raises(AnnotationError, match=r"\(P, 2\)"):
        scan_labelme(spec.root, spec, table)


def test_scan_labelme_rejects_non_list_shapes(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    (root / "A" / "x.json").write_text(
        json.dumps({"shapes": {"label": "cat"}, "imageHeight": 8, "imageWidth": 8}),
        encoding="utf-8",
    )
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="shapes 必须是列表"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_allows_missing_shapes_key(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    (root / "A" / "x.json").write_text(
        json.dumps({"imageHeight": 8, "imageWidth": 8, "imagePath": "x.jpg"}),
        encoding="utf-8",
    )
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert refs[0].boxes.shape == (0, 4)
    assert refs[0].meta["shape_count"] == 0


def test_scan_labelme_rejects_non_mapping_shape(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(root / "A" / "x.json", ["not-a-shape"])
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="必须是映射"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_rejects_non_string_label(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(root / "A" / "x.json", [{"shape_type": "rectangle", "points": [[1, 1], [3, 3]]}])
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="缺少合法的 label"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_rejects_non_string_shape_type(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(
        root / "A" / "x.json", [{"label": "cat", "shape_type": 3, "points": [[1, 1], [3, 3]]}]
    )
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.raises(AnnotationError, match="shape_type 非法"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_rejects_nan_points(tmp_path):
    shapes = [_polygon("cat", [[1.0, float("nan")], [4.0, 1.0], [4.0, 4.0]])]
    spec, table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    with pytest.raises(AnnotationError, match="NaN"):
        scan_labelme(spec.root, spec, table)


def test_scan_labelme_invalid_json_size_fields_fall_back_to_cv2(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "x.jpg")
    _write_labelme(
        root / "A" / "x.json",
        [_rect("cat", 1, 1, 4, 4)],
        height="八",
        width=None,
    )
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    with pytest.warns(UserWarning, match="imageWidth/imageHeight"):
        refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert refs[0].image_size == (8, 8)


def test_scan_labelme_skips_hidden_dirs_and_files(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "A" / "keep.jpg")
    _write_labelme(root / "A" / "keep.json", [_rect("cat", 1, 1, 4, 4)])
    _write_image(root / "A" / ".hidden.jpg")
    _write_labelme(root / "A" / ".hidden.json", [_rect("cat", 1, 1, 4, 4)])
    _write_image(root / ".hidden_dir" / "x.jpg")
    _write_labelme(root / ".hidden_dir" / "x.json", [_rect("cat", 1, 1, 4, 4)])
    spec = _spec(tmp_path, data_type="detection", classes=["cat"], root=root)

    refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert _image_refs(refs) == ["A/keep.jpg"]


def test_scan_labelme_non_recursive(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "top.jpg")
    _write_labelme(root / "top.json", [_rect("cat", 1, 1, 4, 4)])
    _write_image(root / "A" / "nested.jpg")
    _write_labelme(root / "A" / "nested.json", [_rect("cat", 1, 1, 4, 4)])
    spec = _spec(
        tmp_path,
        data_type="detection",
        classes=["cat"],
        root=root,
        parse=ParseConfig(recursive=False),
    )

    refs = scan_labelme(spec.root, spec, ClassTable(["cat"]))

    assert _image_refs(refs) == ["top.jpg"]


def test_scan_labelme_handles_chinese_and_spaces_in_paths(tmp_path):
    root = tmp_path / "data"
    image_path = root / "中文目录" / "a (1).jpg"
    _write_image(image_path)
    _write_labelme(
        root / "中文目录" / "a (1).json", [_rect("猫", 1, 1, 4, 4)], image_name="a (1).jpg"
    )
    spec = _spec(tmp_path, data_type="detection", classes=["猫"], root=root)

    refs = scan_labelme(spec.root, spec, ClassTable(["猫"]))

    assert _image_refs(refs) == ["中文目录/a (1).jpg"]
    assert refs[0].labels.tolist() == [0]


def test_scan_labelme_rejects_non_vision_data_type(tmp_path):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    spec = DVXDatasetSpec(
        version=1,
        data_type="classification",
        layout="labelme",
        root=str(root),
        classes=["cat"],
        tasks=None,
        splits=None,
        parse=ParseConfig(),
    )
    with pytest.raises(DatasetSpecError, match="detection / segmentation"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_scan_labelme_rejects_empty_allowed_shape_types(tmp_path):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    spec = DVXDatasetSpec(
        version=1,
        data_type="detection",
        layout="labelme",
        root=str(root),
        classes=["cat"],
        tasks=None,
        splits=None,
        parse=ParseConfig(labelme_shape_types={"rectangle": "segmentation"}),
    )
    with pytest.raises(DatasetSpecError, match="没有属于"):
        scan_labelme(spec.root, spec, ClassTable(["cat"]))


def test_parse_labelme_file_returns_contract_tuple(tmp_path):
    json_path = tmp_path / "data" / "A" / "x.json"
    _write_labelme(
        json_path,
        [
            _polygon("cat", [[1, 1], [6, 1], [6, 6]]),
            _rect("dog", 1, 2, 5, 6),
            _shape("cat", "circle", [[2, 2], [4, 4], [0, 0]]),
            _polygon("__ignore__", [[0, 0], [1, 0], [1, 1]]),
        ],
    )

    polygons, polygon_labels, rectangle_labels, skipped = parse_labelme_file(
        str(json_path), ClassTable(["cat", "dog"]), IGNORE_LABELS, ("polygon", "rectangle")
    )

    assert len(polygons) == 1
    assert polygons[0].dtype == np.float32
    assert polygons[0].shape == (3, 2)
    assert polygon_labels == [0]
    assert rectangle_labels == [1]
    assert skipped == 1


def test_parse_labelme_file_missing_file(tmp_path):
    with pytest.raises(AnnotationError, match="不存在"):
        parse_labelme_file(
            str(tmp_path / "nope.json"), ClassTable(["cat"]), IGNORE_LABELS, ("polygon",)
        )


# ---------------------------------------------------------------------------
# scan_samples 调度
# ---------------------------------------------------------------------------


def test_scan_samples_classification_layout(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "b.jpg")
    _write_image(root / "cat" / "a.jpg")
    spec = _spec(tmp_path, classes=["cat", "dog"], root=root)

    refs = scan_samples(spec)

    assert _image_refs(refs) == ["cat/a.jpg", "cat/b.jpg"]
    assert [ref.label for ref in refs] == [0, 0]


def test_scan_samples_multitask_aggregates_all_tasks(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "T_licence" / "无" / "a.jpg")
    _write_image(root / "T_logo" / "一体奇瑞标" / "b.jpg")
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[
            ("licence", "T_licence", ["无"]),
            ("logo", "T_logo", ["一体奇瑞标"]),
        ],
        root=root,
    )

    refs = scan_samples(spec)

    assert _image_refs(refs) == ["T_licence/无/a.jpg", "T_logo/一体奇瑞标/b.jpg"]
    assert [ref.task for ref in refs] == ["licence", "logo"]
    assert [ref.label for ref in refs] == [0, 0]


def test_scan_samples_detection_layout(tmp_path):
    shapes = [_rect("cat", 1, 1, 4, 4)]
    spec, _table = _labelme_dataset(tmp_path, shapes)

    refs = scan_samples(spec)

    assert _image_refs(refs) == ["A/x.jpg"]
    assert refs[0].boxes.shape == (1, 4)
    assert refs[0].labels.tolist() == [0]


def test_scan_samples_segmentation_layout(tmp_path):
    shapes = [_polygon("dog", [[1, 1], [5, 1], [5, 5]])]
    spec, _table = _labelme_dataset(tmp_path, shapes, data_type="segmentation")

    refs = scan_samples(spec)

    assert _image_refs(refs) == ["A/x.jpg"]
    assert len(refs[0].segments) == 1
    assert refs[0].segment_labels.tolist() == [1]


def test_scan_samples_does_not_infer_subdir_from_dir_map(tmp_path, capsys):
    """``dir_map`` 不再触发自动子目录收敛：整棵任务目录都会被扫描，也不打印提示。"""
    root = tmp_path / "data"
    _write_image(root / "T_licence" / "无" / "flat.jpg")
    _write_image(root / "T_licence" / "train_data" / "train" / "无" / "train.jpg")
    _write_image(root / "T_licence" / "train_data" / "val" / "无" / "val.jpg")
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[("licence", "T_licence", ["无"])],
        splits=SplitConfig(
            mode="dir", dir_map={"train": "train_data/train", "val": "train_data/val"}
        ),
        root=root,
    )

    refs = scan_samples(spec)
    captured = capsys.readouterr()

    assert _image_refs(refs) == [
        "T_licence/train_data/train/无/train.jpg",
        "T_licence/train_data/val/无/val.jpg",
        "T_licence/无/flat.jpg",
    ]
    assert "仅扫描公共子目录" not in captured.err
    # 扫描范围未被收敛：dir 切分如实报告扁平类目录的样本未被覆盖
    with pytest.raises(SplitError, match="未被 dir_map 覆盖"):
        build_splits(refs, spec.splits, spec)


def test_scan_samples_scans_all_directories_regardless_of_dir_map(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "train" / "cat" / "a.jpg")
    _write_image(root / "val" / "cat" / "b.jpg")
    spec = _spec(
        tmp_path,
        classes=["cat"],
        splits=SplitConfig(mode="dir", dir_map={"train": "train", "val": "val"}),
        root=root,
    )

    refs = scan_samples(spec)

    assert _image_refs(refs) == ["train/cat/a.jpg", "val/cat/b.jpg"]


def test_scan_samples_explicit_subdir_overrides(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "cat" / "flat.jpg")
    _write_image(root / "train_data" / "train" / "cat" / "train.jpg")
    spec = _spec(tmp_path, classes=["cat"], root=root)

    refs = scan_samples(spec, subdir="train_data")

    assert _image_refs(refs) == ["train_data/train/cat/train.jpg"]


def test_scan_samples_rejects_subdir_for_labelme(tmp_path):
    spec, _table = _labelme_dataset(tmp_path, [_rect("cat", 1, 1, 4, 4)])

    with pytest.raises(DatasetSpecError, match="folder_per_class"):
        scan_samples(spec, subdir="train_data")


def test_scan_samples_rejects_duplicate_rel_path(tmp_path):
    root = tmp_path / "data"
    _write_image(root / "T_licence" / "无" / "a.jpg")
    spec = _spec(
        tmp_path,
        data_type="multi_task_classification",
        tasks=[("licence", "T_licence", ["无"]), ("logo", "T_licence", ["无"])],
        root=root,
    )

    with pytest.raises(AnnotationError, match="重复的 rel_path"):
        scan_samples(spec)


def test_scan_samples_validates_spec_root(tmp_path):
    spec = _spec(tmp_path, classes=["cat"])
    spec.root = str(tmp_path / "missing")

    with pytest.raises(DatasetSpecError, match="根目录不存在"):
        scan_samples(spec)


def test_sample_ref_defaults_are_independent():
    first = SampleRef(image_path="/tmp/a.jpg", rel_path="a.jpg")
    second = SampleRef(image_path="/tmp/b.jpg", rel_path="b.jpg")

    assert first.task is None and first.label is None
    assert first.boxes is None and first.labels is None
    assert first.segments is None and first.segment_labels is None
    assert first.ann_path is None and first.image_size is None
    assert first.meta == {}
    assert first.meta is not second.meta
