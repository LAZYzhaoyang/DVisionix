# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: P4-A2 预测器测试（toy 三模型端到端 + checkpoint 加载 + 状态恢复）。
"""P4-A2 预测器测试。

覆盖契约 ``docs/design/v2_p4_contract.md`` §3 与 §6 中 A2 的验收项：

- toy 三模型（``SimpleCNN`` / ``GridDetectionModel`` / ``SimpleSegmentationModel``）
  端到端跑通 ``predict``，不依赖任何真实数据集；
- 检测框全部落在原图范围内，且反变换结果与「直接 decode 后按 scale 缩放」一致；
- 分割 mask 形状等于原图，且上采样方式确为最近邻；
- 分类 probs 和为 1、topk 按概率降序且与 probs 对应；
- ``from_checkpoint`` 正常往返（纯 state_dict / Trainer 存档 / 包装前缀），
  损坏或结构不匹配时抛 ``CheckpointError``，完整 Trainer 存档需要 ``trusted=True``；
- 预测结束后模型原来的 ``training`` 状态被恢复（含前向抛异常的情况）。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dvisionix.errors import CheckpointError, ConfigError
from dvisionix.inference.predictors import (
    BasePredictor,
    ClassificationPredictor,
    DetectionPredictor,
    SegmentationPredictor,
    build_predictor,
)
from dvisionix.models.toy import GridDetectionModel, SimpleCNN, SimpleSegmentationModel

#: 图片尺寸（H, W）都刻意选非正方形且不被预处理尺寸整除，便于暴露坐标/插值错误
IMAGE_HW = (37, 53)


class _OpaqueState:
    """模拟 Trainer checkpoint 里的非张量状态（RNG / 优化器对象）。

    ``weights_only=True`` 的反序列化白名单只允许张量与基本容器，自定义对象会被
    拒绝；用它验证「完整 Trainer checkpoint 需要 trusted=True」的提示信息。
    """

    def __init__(self, payload: str = "rng-state") -> None:
        self.payload = payload


class _BatchAwareModel(nn.Module):
    """``needs_batch=True`` 的最小分类模型：记录 forward 收到的 ``batch``。"""

    needs_batch = True

    def __init__(self) -> None:
        super().__init__()
        self.received_batch = None
        self.linear = nn.Linear(3 * 32 * 32, 4)

    def forward(self, x, batch=None):
        """记录 batch 后返回 logits。"""
        self.received_batch = batch
        return self.linear(x.flatten(1))


class _ExplodingModel(nn.Module):
    """前向必定抛异常的最小模型（验证异常路径下的 training 状态恢复）。"""

    def forward(self, x, **kwargs):
        """直接抛 RuntimeError。"""
        raise RuntimeError("boom")


def _make_image(size=IMAGE_HW) -> np.ndarray:
    """生成确定性的 uint8 RGB 测试图（HWC）。"""
    h, w = size
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _reset_seed() -> None:
    """每个测试前固定随机种子，保证 toy 模型初始化与输入可复现。"""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def classification_predictor() -> ClassificationPredictor:
    """``SimpleCNN`` 分类预测器（32x32 输入是 SimpleCNN 的固定展平尺寸）。"""
    return ClassificationPredictor(
        SimpleCNN(num_classes=3),
        "classification",
        device="cpu",
        image_size=32,
        topk=3,
    )


@pytest.fixture
def detection_predictor() -> DetectionPredictor:
    """``GridDetectionModel`` 检测预测器（低阈值保证能拿到非空框）。"""
    return DetectionPredictor(
        GridDetectionModel(num_classes=3),
        "detection",
        device="cpu",
        image_size=64,
        score_threshold=0.01,
    )


@pytest.fixture
def segmentation_predictor() -> SegmentationPredictor:
    """``SimpleSegmentationModel`` 分割预测器。"""
    return SegmentationPredictor(
        SimpleSegmentationModel(num_classes=3),
        "segmentation",
        device="cpu",
        image_size=64,
    )


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------
def test_classification_returns_probs_and_sorted_topk(classification_predictor):
    """分类：probs 形状/和为 1，topk 降序且与 probs 一一对应。"""
    res = classification_predictor.predict(_make_image())

    assert res.task_type == "classification"
    assert res.orig_size == IMAGE_HW
    assert res.probs is not None
    assert res.probs.shape == (3,)
    assert np.isclose(res.probs.sum(), 1.0, atol=1e-6)

    assert res.topk is not None
    assert len(res.topk) == 3
    indices = [index for index, _ in res.topk]
    values = [value for _, value in res.topk]
    assert values == sorted(values, reverse=True)
    assert np.allclose([res.probs[index] for index in indices], values, atol=1e-6)

    # 分类结果不应包含检测/分割字段
    assert res.boxes is None and res.scores is None and res.labels is None
    assert res.mask is None


def test_classification_topk_truncated_to_num_classes():
    """分类：topk 大于类别数时截断为 num_classes。"""
    predictor = ClassificationPredictor(
        SimpleCNN(num_classes=2), "classification", device="cpu", image_size=32, topk=5
    )
    assert len(predictor.predict(_make_image()).topk) == 2


def test_classification_invalid_topk_raises_config_error():
    """分类：topk < 1 是非法配置。"""
    with pytest.raises(ConfigError):
        ClassificationPredictor(
            SimpleCNN(num_classes=3), "classification", device="cpu", image_size=32, topk=0
        )


def test_classification_class_names_written_to_meta(classification_predictor):
    """分类：class_names 写入 Results.meta，供可视化/JSON 使用。"""
    predictor = ClassificationPredictor(
        SimpleCNN(num_classes=3),
        "classification",
        device="cpu",
        image_size=32,
        class_names=["cat", "dog", "bird"],
    )
    assert predictor.predict(_make_image()).meta["class_names"] == ["cat", "dog", "bird"]


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------
def test_detection_boxes_are_inside_original_image(detection_predictor):
    """检测：框反变换回原图坐标，且全部落在 [0, W-1] x [0, H-1]。"""
    res = detection_predictor.predict(_make_image())

    assert res.task_type == "detection"
    assert res.orig_size == IMAGE_HW
    assert res.boxes is not None
    assert res.boxes.ndim == 2 and res.boxes.shape[1] == 4
    assert res.boxes.shape[0] > 0
    assert res.scores is not None and res.scores.shape == (res.boxes.shape[0],)
    assert res.labels is not None and res.labels.shape == (res.boxes.shape[0],)
    assert res.probs is None and res.mask is None

    orig_h, orig_w = IMAGE_HW
    assert res.boxes[:, 0::2].min() >= 0.0
    assert res.boxes[:, 0::2].max() <= float(orig_w - 1)
    assert res.boxes[:, 1::2].min() >= 0.0
    assert res.boxes[:, 1::2].max() <= float(orig_h - 1)


def test_detection_reverse_transform_matches_manual_decode(detection_predictor):
    """检测：反变换等于「处理后坐标 / scale」，scale = 处理后尺寸 / 原图尺寸。"""
    image = _make_image()
    tensor, meta = detection_predictor.preprocess(image)
    processed_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))

    model = detection_predictor.model
    model.eval()
    with torch.no_grad():
        preds = model(tensor)
        boxes_p, scores_p, labels_p = model.decode(
            preds,
            processed_hw,
            score_threshold=detection_predictor.score_threshold,
            iou_threshold=detection_predictor.iou_threshold,
            max_detections=detection_predictor.max_detections,
        )
    res = detection_predictor.predict(image)

    sh, sw = meta.scale
    assert sw == pytest.approx(processed_hw[1] / IMAGE_HW[1])
    assert sh == pytest.approx(processed_hw[0] / IMAGE_HW[0])
    expected = boxes_p[0].numpy() / np.array([sw, sh, sw, sh], dtype=np.float32)
    expected[:, 0::2] = np.clip(expected[:, 0::2], 0.0, float(IMAGE_HW[1] - 1))
    expected[:, 1::2] = np.clip(expected[:, 1::2], 0.0, float(IMAGE_HW[0] - 1))
    assert np.allclose(res.boxes, expected, atol=1e-4)
    assert np.allclose(res.scores, scores_p[0].numpy(), atol=1e-6)
    assert np.array_equal(res.labels, labels_p[0].numpy())


def test_detection_empty_result_keeps_shapes():
    """检测：阈值过高时返回空数组而不是 None 或崩溃。"""
    predictor = DetectionPredictor(
        GridDetectionModel(num_classes=3),
        "detection",
        device="cpu",
        image_size=64,
        score_threshold=0.9999,
    )
    res = predictor.predict(_make_image())

    assert res.boxes is not None and res.boxes.shape == (0, 4)
    assert res.scores is not None and res.scores.shape == (0,)
    assert res.labels is not None and res.labels.shape == (0,)


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------
def test_segmentation_mask_has_original_size(segmentation_predictor):
    """分割：mask 形状 = 原图尺寸，dtype=int64，取值在类别范围内。"""
    res = segmentation_predictor.predict(_make_image())

    assert res.task_type == "segmentation"
    assert res.orig_size == IMAGE_HW
    assert res.mask is not None
    assert res.mask.shape == IMAGE_HW
    assert res.mask.dtype == np.int64
    assert set(np.unique(res.mask)) <= {0, 1, 2}
    assert res.boxes is None and res.probs is None
    assert "probs" not in res.meta


def test_segmentation_mask_uses_nearest_upsampling(segmentation_predictor):
    """分割：mask 是处理后 argmax 的最近邻上采样（不是双线性/混合值）。"""
    image = _make_image()
    res = segmentation_predictor.predict(image)

    tensor, _ = segmentation_predictor.preprocess(image)
    model = segmentation_predictor.model
    model.eval()
    with torch.no_grad():
        small = model(tensor).argmax(dim=1)[0].numpy()
    small_h, small_w = small.shape
    orig_h, orig_w = IMAGE_HW

    # F.interpolate(mode="nearest") 的取源规则：src = floor(dst * src_size / dst_size)
    ys = np.floor(np.arange(orig_h) * small_h / orig_h).astype(np.int64)
    xs = np.floor(np.arange(orig_w) * small_w / orig_w).astype(np.int64)
    expected = small[np.minimum(ys, small_h - 1)[:, None], np.minimum(xs, small_w - 1)[None, :]]
    assert np.array_equal(res.mask, expected)


def test_segmentation_return_probs_writes_meta():
    """分割：return_probs=True 时把原图尺寸的 softmax 概率写入 Results.meta。"""
    predictor = SegmentationPredictor(
        SimpleSegmentationModel(num_classes=3),
        "segmentation",
        device="cpu",
        image_size=64,
        return_probs=True,
    )
    res = predictor.predict(_make_image())

    probs = res.meta["probs"]
    assert probs.shape == (3, IMAGE_HW[0], IMAGE_HW[1])
    assert probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 通用行为：predict_many / needs_batch / eval 状态恢复 / build_predictor
# ---------------------------------------------------------------------------
def test_predict_many_returns_one_result_per_input(segmentation_predictor):
    """predict_many：逐个处理，结果数量与顺序与输入一致。"""
    images = [_make_image((20, 30)), _make_image((40, 50))]
    results = segmentation_predictor.predict_many(images)

    assert len(results) == 2
    assert [result.orig_size for result in results] == [(20, 30), (40, 50)]
    assert all(isinstance(result.mask, np.ndarray) for result in results)


def test_needs_batch_model_receives_batch_dict():
    """needs_batch 模型：按 DetectionTask 的约定收到 batch={"image": images}。"""
    model = _BatchAwareModel()
    predictor = ClassificationPredictor(model, "classification", device="cpu", image_size=32)
    res = predictor.predict(_make_image())

    assert model.received_batch is not None
    assert set(model.received_batch) == {"image"}
    assert model.received_batch["image"].shape == (1, 3, 32, 32)
    assert res.probs is not None and res.probs.shape == (4,)


@pytest.mark.parametrize(
    ("model_factory", "task_type", "image_size"),
    [
        (lambda: SimpleCNN(num_classes=3), "classification", 32),
        (lambda: GridDetectionModel(num_classes=3), "detection", 64),
        (lambda: SimpleSegmentationModel(num_classes=3), "segmentation", 64),
    ],
    ids=["classification", "detection", "segmentation"],
)
def test_predict_restores_training_mode(model_factory, task_type, image_size):
    """三种预测器都必须恢复模型原来的 training 状态（train / eval 两种起点）。"""
    model = model_factory()
    predictor = build_predictor(model, task_type, device="cpu", image_size=image_size)

    assert model.training is True
    predictor.predict(_make_image())
    assert model.training is True

    model.eval()
    predictor.predict(_make_image())
    assert model.training is False


def test_predict_restores_training_mode_after_forward_error():
    """前向抛异常时也必须恢复 training 状态（finally 分支）。"""
    model = _ExplodingModel()
    predictor = ClassificationPredictor(model, "classification", device="cpu", image_size=32)
    model.train()
    with pytest.raises(RuntimeError, match="boom"):
        predictor.predict(_make_image())
    assert model.training is True


@pytest.mark.parametrize(
    ("model_factory", "task_type", "predictor_cls", "image_size"),
    [
        (lambda: SimpleCNN(num_classes=3), "classification", ClassificationPredictor, 32),
        (lambda: GridDetectionModel(num_classes=3), "detection", DetectionPredictor, 64),
        (lambda: SimpleSegmentationModel(num_classes=3), "segmentation", SegmentationPredictor, 64),
    ],
    ids=["classification", "detection", "segmentation"],
)
def test_build_predictor_returns_matching_subclass(
    model_factory, task_type, predictor_cls, image_size
):
    """build_predictor：按任务类型返回对应子类（大小写/空白不敏感）。"""
    predictor = build_predictor(model_factory(), f" {task_type.upper()} ", device="cpu")
    assert isinstance(predictor, predictor_cls)
    assert isinstance(predictor, BasePredictor)
    assert predictor.task_type == task_type
    assert predictor.image_size == 224  # 默认值，未显式传 image_size


def test_build_predictor_forwards_task_kwargs():
    """build_predictor：子类参数（topk）透传。"""
    predictor = build_predictor(
        SimpleCNN(num_classes=3), "classification", device="cpu", image_size=32, topk=2
    )
    assert isinstance(predictor, ClassificationPredictor)
    assert predictor.topk == 2
    assert len(predictor.predict(_make_image()).topk) == 2


def test_build_predictor_unsupported_task_raises_config_error():
    """build_predictor：不支持的任务类型抛 ConfigError。"""
    with pytest.raises(ConfigError) as excinfo:
        build_predictor(SimpleCNN(num_classes=3), "simclr", device="cpu")
    assert "不支持的推理任务类型" in str(excinfo.value)


# ---------------------------------------------------------------------------
# from_checkpoint
# ---------------------------------------------------------------------------
def test_from_checkpoint_loads_plain_state_dict(tmp_path):
    """from_checkpoint：纯 state_dict 往返，权重逐键一致。"""
    source = SimpleCNN(num_classes=3)
    path = tmp_path / "plain.pt"
    torch.save(source.state_dict(), path)

    loaded = SimpleCNN(num_classes=3)
    predictor = ClassificationPredictor.from_checkpoint(
        loaded, str(path), "classification", image_size=32, device="cpu"
    )
    assert isinstance(predictor, ClassificationPredictor)
    assert predictor.model is loaded
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_from_checkpoint_reads_model_state_dict_key(tmp_path):
    """from_checkpoint：Trainer 存档自动取 model_state_dict，忽略其它键。"""
    source = GridDetectionModel(num_classes=3)
    path = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": source.state_dict(), "epoch": 7, "best": 0.5}, path)

    loaded = GridDetectionModel(num_classes=3)
    DetectionPredictor.from_checkpoint(loaded, str(path), "detection", image_size=64, device="cpu")
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_from_checkpoint_strips_ddp_and_compile_prefixes(tmp_path):
    """from_checkpoint：剥离 module.（DDP）与 _orig_mod.（torch.compile）前缀。"""
    source = SimpleSegmentationModel(num_classes=3)
    wrapped = {f"module._orig_mod.{key}": value for key, value in source.state_dict().items()}
    path = tmp_path / "wrapped.pt"
    torch.save(wrapped, path)

    loaded = SimpleSegmentationModel(num_classes=3)
    SegmentationPredictor.from_checkpoint(
        loaded, str(path), "segmentation", image_size=64, device="cpu"
    )
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_from_checkpoint_full_checkpoint_requires_trusted(tmp_path):
    """from_checkpoint：完整 Trainer 存档在安全模式下失败，提示 trusted=True。"""
    source = SimpleCNN(num_classes=3)
    path = tmp_path / "full.pt"
    torch.save(
        {"model_state_dict": source.state_dict(), "rng_state": _OpaqueState(), "epoch": 3},
        path,
    )

    with pytest.raises(CheckpointError) as excinfo:
        ClassificationPredictor.from_checkpoint(
            SimpleCNN(num_classes=3), str(path), "classification", image_size=32, device="cpu"
        )
    assert "trusted=True" in str(excinfo.value)

    # 显式信任后可以正常加载（非张量状态被忽略，只取 model_state_dict）
    loaded = SimpleCNN(num_classes=3)
    ClassificationPredictor.from_checkpoint(
        loaded,
        str(path),
        "classification",
        trusted=True,
        image_size=32,
        device="cpu",
    )
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_from_checkpoint_structure_mismatch_raises(tmp_path):
    """from_checkpoint：键/形状不匹配时 strict 加载报 CheckpointError。"""
    path = tmp_path / "mismatch.pt"
    torch.save(SimpleCNN(num_classes=5).state_dict(), path)
    with pytest.raises(CheckpointError) as excinfo:
        ClassificationPredictor.from_checkpoint(
            SimpleCNN(num_classes=3), str(path), "classification", image_size=32, device="cpu"
        )
    assert "结构不匹配" in str(excinfo.value)


def test_from_checkpoint_corrupt_file_raises(tmp_path):
    """from_checkpoint：损坏文件抛 CheckpointError（并提示可信来源/trusted）。"""
    path = tmp_path / "broken.pt"
    path.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(CheckpointError) as excinfo:
        ClassificationPredictor.from_checkpoint(
            SimpleCNN(num_classes=3), str(path), "classification", image_size=32, device="cpu"
        )
    assert "trusted=True" in str(excinfo.value)


def test_from_checkpoint_missing_file_raises(tmp_path):
    """from_checkpoint：文件不存在抛 CheckpointError。"""
    path = tmp_path / "nope.pt"
    with pytest.raises(CheckpointError) as excinfo:
        ClassificationPredictor.from_checkpoint(
            SimpleCNN(num_classes=3), str(path), "classification", image_size=32, device="cpu"
        )
    assert "不存在" in str(excinfo.value)


def test_from_checkpoint_non_state_dict_raises(tmp_path):
    """from_checkpoint：checkpoint 内容不是 state_dict 时抛 CheckpointError。"""
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(CheckpointError) as excinfo:
        ClassificationPredictor.from_checkpoint(
            SimpleCNN(num_classes=3), str(path), "classification", image_size=32, device="cpu"
        )
    assert "state_dict" in str(excinfo.value)
