# DVisionix v2 —— P2 实现契约（Lead 制定，实现方必须遵守）

> 状态：**生效**。本文件是 P2 阶段所有实现的唯一接口事实来源。
> 环境：统一使用 `/data/lzy/Env/condaEnvs/dvisionix/bin/python`（`$PY`）。
> 风格：UTF-8；中文 docstring；ruff line-length 100（`E,F,W,I`）、black 100；不得新增第三方依赖。

---

## 0. P2 目标（来自已确认的规划）

1. **任务/模型/数据契约**：不允许"分割任务配分类模型"这类静默错配，必须给出可操作的错误。
2. **Task 承担任务知识**：预设变换由 Task 提供，`tools/train.py` 不再用 if/else 硬编码任务分派。
3. **DVX 数据集接入训练**：`data.dataset: dvx` + `data.spec` 即可训练分类/检测/分割。
4. **DDP 修复**：`strategy: auto` 正确识别 torchrun；多卡验证的跨设备张量聚合不再崩溃。
5. **测试基线转绿**：全量 pytest **0 failed**（含此前 4 个环境性失败）。
6. 真实数据验收：cls / det / seg 各训练 1 epoch。

**不在 P2 范围**：多任务分类模型与任务（P3）、检索训练、`Task.build_dataset` 完整接管数据装配、推理与导出。

---

## 1. P2-1 模型能力声明（`dvisionix/models/base.py`）

```python
class BaseModel(nn.Module):
    # 新增：模型支持的任务能力集合；默认空表示"由 task_type 推导"
    supported_tasks: ClassVar[tuple[str, ...]] = ()
```

新增模块级函数（必须导出）：

```python
def model_task_types(model: nn.Module) -> tuple[str, ...]:
    """返回模型声明的任务能力：优先 supported_tasks，其次 task_type，均无返回 ()。"""
```

- **不修改** `TASK_TYPES = {"classification","detection","segmentation"}` 的既有语义。
- 不改动任何既有模型的构造行为；仅新增声明能力。
- 建议：为 `FeatureBackboneBase` 等非任务模型保持 `()`（它们不是完整模型）。

## 2. P2-2 Task 契约（`dvisionix/training/tasks/base.py`）

```python
class BaseTask(ABC):
    task_type: ClassVar[str] = ""
    accepted_model_tasks: ClassVar[tuple[str, ...]] = ()
    accepted_data_types: ClassVar[tuple[str, ...]] = ()
    required_batch_fields: ClassVar[tuple[str, ...]] = ("image",)

    @classmethod
    def build_transforms(cls, train: bool, image_size: int):
        """返回该任务的预设 TransformPipeline；默认 None（表示不提供）。"""
        return None
```

**内置任务声明表（必须逐条实现）**：

| Task 类 | `task_type` | `accepted_model_tasks` | `accepted_data_types` | `required_batch_fields` |
|---|---|---|---|---|
| `ClassificationTask` | `classification` | `("classification",)` | `("classification",)` | `("image","label")` |
| `MultiLabelTask` | `multi_label` | `("classification",)` | `("multi_label",)` | `("image","label")` |
| `DetectionTask` | `detection` | `("detection",)` | `("detection",)` | `("image","boxes","labels")` |
| `SegmentationTask` | `segmentation` | `("segmentation",)` | `("segmentation",)` | `("image","mask")` |
| `MaskFormerTask` | `segmentation` | `("segmentation",)` | `("segmentation","instance_segmentation","panoptic_segmentation")` | `("image","mask")` |
| `SimCLRTask` | `simclr` | `("classification",)` | `("simclr",)` | `("image1","image2")` |
| `LinearEvalTask` | `linear_eval` | `("classification",)` | `("classification",)` | `("image","label")` |

`build_transforms` 映射（函数内 import，避免循环依赖）：

| Task | 返回 |
|---|---|
| `ClassificationTask` / `MultiLabelTask` / `LinearEvalTask` | `ClassificationTransforms(train=train, image_size=image_size)` |
| `DetectionTask` | `DetectionTransforms(...)` |
| `SegmentationTask` / `MaskFormerTask` | `SegmentationTransforms(...)` |
| `SimCLRTask` | `SimCLRTransforms(...)` |

## 3. P2-3 契约校验（新增 `dvisionix/training/contract.py`）

```python
def check_task_model(task, model, dataset=None) -> None: ...
def check_task_data(task, dataset) -> None: ...
def check_batch_fields(task, batch) -> None: ...
```

规则：

1. **check_task_model**
   - 模型能力 `model_task_types(model)` 与 `task.accepted_model_tasks` 交集为空 → `TaskModelMismatchError`。
   - 若 `dataset` 有 `num_classes` 且 `model` 有 `num_classes` 且两者不相等 → `TaskDataMismatchError`。
   - 消息必须包含：任务类名、模型类名、模型声明的能力、任务要求的能力、以及"应如何修复"的 hint。
2. **check_task_data**
   - `dataset` 没有 `data_type` 属性（非 DVX 数据集，如 CustomDataset/presets）→ 直接放行（不报错）。
   - `data_type` 不在 `task.accepted_data_types` → `TaskDataMismatchError`，消息含 dataset 的 data_type、任务要求的集合。
3. **check_batch_fields**
   - `required_batch_fields` 中任一键不在 batch → `TaskDataMismatchError`，消息含缺失字段、实际 batch 键、以及"检查 data_type 与 transforms/collate"的 hint。
   - 只在**第一个训练 batch**检查一次（Trainer 内部用标志位），避免每步开销。

> **渐进式契约（Lead 裁定）**：任务若未声明 `accepted_model_tasks` / `accepted_data_types`
> （保持默认空元组，常见于用户自定义任务），则跳过对应校验，避免默认规则误伤既有自定义任务；
> `num_classes` 校验不受影响、仍会执行。内置 7 个任务均已显式声明，
> 因此"分割任务配分类模型"仍会在训练开始前被拦下。

**Trainer 接线**（`training/trainer.py`）：
- `fit()` 开始处（`self.model = self._wrap_model(model)` 之前）调用
  `check_task_model(self.task, model, self.train_loader.dataset)` 与
  `check_task_data(self.task, self.train_loader.dataset)`。
- 训练循环第一个 batch 调用 `check_batch_fields`（每 epoch 不必重复，整个 fit 一次即可）。
- 校验失败必须**在训练开始前**抛出，不允许训练到一半才报。

## 4. P2-4 DDP 修复（`training/trainer.py`）

### 4.1 `strategy: auto` 识别 torchrun
`_init_distributed()` 中 `auto` 判定改为（满足任一即 `ddp`）：
1. 进程组已初始化且 `world_size > 1`（现有行为）；或
2. 环境变量 `RANK` 与 `WORLD_SIZE` 同时存在，且 `int(WORLD_SIZE) > 1`（torchrun 启动）。
否则 `none`。判定后保持现有 `ddp` 分支（缺 env 时给出原有报错）。

### 4.2 跨设备聚合
根因：`all_gather_object` 收集到的张量仍停留在各自 GPU，`_concat_objects` 里 `torch.cat` 跨设备失败。

要求：
- 聚合前把 `preds` / `targets` 递归转成 **CPU** 张量再 `all_gather_object`；
- 合并（`_concat_objects`）完成后，把结果递归搬回 `self.device` 再 `task.update_metrics`；
- 保持 `gather_report()` 统计口径不变；单进程路径行为不变；
- 递归工具需支持 `Tensor / dict / list / tuple`（检测的 `(boxes_list, scores_list, labels_list)`）。

### 4.3 测试基线转绿
- `tests/test_training/test_ddp_smoke.py`：新增"必须由 torchrun 启动"的 `skipif`
  （`RANK`/`WORLD_SIZE` 缺失即 skip，理由说明用 torchrun 运行）。保留 2+ GPU 守卫。
- `tests/test_metrics/test_metrics.py::_cls_logits`：当前 one-hot×10 使非目标类**全部并列 0**，
  `torch.topk` 平票顺序未定义导致断言随 torch 版本漂移。改为非目标类**互不相等**的确定值
  （例如减去 `arange(num_classes) * 1e-3`），保持 argmax 与既有 accuracy=0.8 断言不变，
  使 TopK 断言确定成立。
- 目标：`pytest tests -q` → **0 failed**。

## 5. P2-5 DVX 数据接入（`tools/train.py`，Wave 2）

- 构造顺序改为：`config → task → transforms(task) → dataset → model → trainer`。
- 新增数据源：`data.dataset: "dvx"` + `data.spec: <dvx_dataset.yaml 路径>`：
  - `train_ds = DVXDataset(spec, split="train", transforms=task.build_transforms(True, image_size))`
  - `val_ds = DVXDataset(spec, split="val", transforms=...train=False)`
  - `DataLoader(collate_fn=getattr(ds, "collate_fn", None))`
- 保留 `synthetic` 与公开数据集预设两条既有路径（不删除）。
- `build_transforms(cfg)` 改为委托 `task.build_transforms`；若任务返回 None 则报明确错误。
- 新增 `configs/examples/dvx_{classification,detection,segmentation}.yaml`（示例模板，
  `data.spec` 用占位路径），并在 `tests/test_e2e_configs.py` 的
  `NON_TRAINING_DIR_PREFIXES` 加入 `"examples/"`（外部数据配置不参与 CI 门禁）。

## 6. 验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python
$PY -m pytest tests -q                      # 0 failed
$PY -m pytest tests/test_e2e_configs.py -q  # 24 passed
$PY -m ruff check dvisionix tools tests
$PY -m black --check dvisionix tools tests
# DDP（真实 2 卡）
torchrun --nproc_per_node=2 -m pytest tests/test_training/test_ddp_smoke.py -q
# 真实数据 1 epoch（Lead 执行）
$PY tools/train.py --config configs/examples/dvx_classification.yaml --cfg-options data.spec=/data/lzy/DVisionix/project/data/cls/dvx_dataset.yaml --work-dir /tmp/p2_cls
$PY tools/train.py --config configs/examples/dvx_detection.yaml      --cfg-options data.spec=/data/lzy/DVisionix/project/data/det/dvx_dataset.yaml --work-dir /tmp/p2_det
$PY tools/train.py --config configs/examples/dvx_segmentation.yaml   --cfg-options data.spec=/data/lzy/DVisionix/project/data/seg/dvx_dataset.yaml --work-dir /tmp/p2_seg
```

## 7. 文件所有权

| 归属 | 文件 |
|---|---|
| Wave 1 实现（training-core） | `dvisionix/models/base.py`、`dvisionix/training/{contract.py(新增),trainer.py,builder.py}`、`dvisionix/training/tasks/*.py`、`tests/test_training/test_contracts.py(新增)`、`tests/test_training/test_ddp_smoke.py`、`tests/test_training/test_ddp_cpu.py`、`tests/test_metrics/test_metrics.py` |
| Wave 2 实现（wiring） | `tools/train.py`、`configs/examples/*`、`tests/test_e2e_configs.py` |
| Lead | 本契约、审查、DDP/真实数据验收、git 提交 |
| 文档 Wave | `docs/dataset_spec.md`、`docs/quick_start.md`、`README.md`、`CHANGELOG.md` |
