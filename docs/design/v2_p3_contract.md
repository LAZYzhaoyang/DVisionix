# DVisionix v2 —— P3 实现契约（Lead 制定，实现方必须遵守）

> 状态：**生效**。P3 = 路由式多任务分类。环境：`PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python`。
> 风格：UTF-8；中文 docstring；ruff line-length 100 / black 100；不得新增第三方依赖。

---

## 0. P3 目标

在 `multicls` 数据集（`data_type: multi_task_classification`，结构 `<任务>/<类别>/*.jpg`）上跑通
**共享 backbone + 每任务一个分类头**的训练：每个样本按自身 `task` 路由到对应头，各任务独立算损失与指标。

**不在 P3 范围**：检索训练、多任务模型的推理/导出专项、跨任务共享标签空间。

## 1. 模型：`MultiTaskClassifier`（新增 `dvisionix/models/classifiers/multi_task.py`）

```python
@MODELS.register()
@MODELS.register(name="multi_task_classifier")
class MultiTaskClassifier(BaseModel):
    #: 类属性（不是实例），供 training.contract 的能力校验读取
    supported_tasks = ("multi_task_classification",)

    def __init__(self, backbone: dict, heads: dict[str, dict], **kwargs) -> None: ...
    # 只读属性
    task_names: tuple[str, ...]          # heads 的键，顺序稳定（dict 插入序）
    num_classes: dict[str, int]          # {task: head.num_classes}（property，不能用 None）

    def forward(self, x, task: list[str] | None = None, **kwargs):
        """task is None -> {task_name: logits_all}（每头对整批算）；
        task 给定时按样本路由：只对属于该任务的子集调用对应头，返回 {task_name: (n_i, C_i)}。
        必须只做一次 backbone 前向（用索引子集切特征），不得为每个任务重复过 backbone。"""
```

- backbone 用 `BACKBONES.build`，并 `setdefault("features_only", False)`；输入特征维度取
  `backbone.num_features`（或 `out_channels[-1]` 兜底）。
- 每个 head 注入 `in_channels`；head 配置由调用方给出（含 `num_classes`）。
- 注册到 `MODELS`；在 `dvisionix/models/classifiers/__init__.py` 与
  `dvisionix/models/__init__.py` 中导出（只加不改）。

## 2. 任务：`MultiTaskClassificationTask`（新增 `dvisionix/training/tasks/multi_task_classification.py`）

```python
class MultiTaskClassificationTask(BaseTask):
    task_type = "multi_task_classification"
    accepted_model_tasks = ("multi_task_classification",)
    accepted_data_types = ("multi_task_classification",)
    required_batch_fields = ("image", "label", "task")

    @classmethod
    def build_transforms(cls, train: bool, image_size: int):  # 返回 ClassificationTransforms
        ...

    def __init__(self, num_classes: dict | None = None, task_loss_weights: dict[str,float] | None = None,
                 learning_rate=None, weight_decay=None, optimizer_cfg=None, scheduler_cfg=None,
                 loss=None, metrics=None) -> None: ...
```

行为要求：
1. **样本结构**：`batch["image"] (B,C,H,W)`、`batch["label"] (B,) long`、`batch["task"] list[str]`（长度 B）。
2. **训练步**：
   - 调用 `model(images, task=task_list)` 得到 `{task: logits}`；
   - 对每个出现过的任务用**该任务的子集**算交叉熵（默认共享一个 `CrossEntropy`；`task_loss_weights` 可调权重）；
   - 返回 `{"loss": 加权总损失, "<task>_loss": ..., "<task>_acc": ...}`（键随本 batch 实际任务变化）。
3. **验证步**：返回
   `{"loss": ..., "preds": {task: logits}, "targets": {task: labels}}`，
   且 **preds/targets 的键必须恒等于模型的全部任务名**（本 batch 未出现的任务用
   `(0, C_task)` 空张量与 `(0,)` 空标签），保证 DDP `all_gather` 时各 rank 的 dict 键一致。
4. **指标**：每个任务一个 `MetricCollection`（在**该任务第一次出现时**按 `C=logits.shape[1]` 惰性创建
   `get_preset_metrics("classification", num_classes=C)`）；`update_metrics` 跳过空张量；
   `on_validation_epoch_end` 返回**带任务前缀**的键（如 `licence_accuracy`、`logo_f1`）并 reset；
   `reset_metrics` 重置全部任务集合。
5. 任务名全集来源：优先 `getattr(model, "task_names", None)`，否则取本 batch 的 task 去重排序。
6. 在 `dvisionix/training/tasks/__init__.py` 的注册元组中加入本任务（`build_task` 可构建）。

## 3. 契约扩展

### 3.1 `dvisionix/data/dvx_dataset.py`
新增只读属性 `task_num_classes: dict[str, int] | None`：
- `multi_task_classification` → `{task.name: len(task.classes)}`；
- 其它 data_type → `None`。

### 3.2 `dvisionix/training/contract.py`
`check_task_model` 增加**字典类别数**校验（保留现有标量逻辑不变）：

```python
ds_by_task = getattr(dataset, "task_num_classes", None)
model_nc = getattr(model, "num_classes", None)
if isinstance(ds_by_task, dict) and isinstance(model_nc, dict):
    # 任务名集合一致 + 每个任务类别数一致，否则 TaskDataMismatchError（消息列出差异）
```

## 4. 训练入口接线（`tools/train.py` + `config/schema.py`）

1. `dvisionix/config/schema.py::TASK_TYPES` 增加 `"multi_task_classification"`。
2. `tools/train.py::_TASK_MAPPING` 增加 `"multi_task_classification": "MultiTaskClassificationTask"`
   （`tests/test_config/test_schema.py::test_task_mapping_covers_all_task_types` 要求两者集合完全一致）。
3. `tools/train.py::main()` 的必填校验按任务类型放开：`multi_task_classification` **不要求**
   `model.num_classes`（其余任务仍要求）。
4. `build_task_cfg`：多任务时**不注入** `num_classes`（模型按 heads 自带类别数）；
   `cfg.model.num_classes` 的访问不得无条件执行（当前 `build_data` 第 156 行也需改为按需读取）。
5. `build_data` 的 dvx 分支行为不变；`synthetic` 分支保持原样。
6. 新增 `configs/examples/dvx_multi_task.yaml`：
   - `task_type: multi_task_classification`；`model.type: multi_task_classifier`；
     backbone 用内置 `mobilenetv3_backbone`；heads：`licence`（3 类）、`logo`（2 类）；
   - `task.type: MultiTaskClassificationTask`；
   - `data: {dataset: dvx, spec: ./data/multicls/dvx_dataset.yaml, image_size: 224}`；
   - `training: {num_epochs: 1, batch_size: 16, amp: false, num_workers: 0, early_stopping: {enabled: false}}`；
   - `loss: null`（任务内部创建 CE）、`metrics: null`。
   - 该目录已在 E2E `NON_TRAINING_DIR_PREFIXES` 中排除，无需重复登记。

## 5. 测试

- `tests/test_models/test_multi_task.py`（新增）：路由正确性（每个任务只对子集调用头）、
  `task=None` 时返回全部头输出、`num_classes` 为 dict、`task_names` 稳定。
- `tests/test_training/test_multi_task_task.py`（新增）：训练步 loss/各任务键；验证步
  preds/targets 键全集；惰性指标按任务前缀汇总；空任务 batch 不报错。
- `tests/test_training/test_contracts.py`（扩展）：`task_num_classes` 与模型 dict 类别数不一致
  报 `TaskDataMismatchError`；缺失任务名也报错。
- `tests/test_training/test_dvx_config_smoke.py`（扩展，若方便）：多任务 mini 数据集装配出
  `batch["task"]` 为 list[str]，模型能按 task 路由。

## 6. 验收

```bash
PY=/data/lzy/Env/condaEnvs/dvisionix/bin/python
$PY -m pytest tests -q                     # 0 failed（基线 903 passed / 1 skipped）
$PY -m pytest tests/test_e2e_configs.py -q # 24 passed
$PY -m ruff check dvisionix tools tests && $PY -m black --check dvisionix tools tests
# 真实 multicls 1 epoch（Lead 执行）
$PY tools/train.py --config configs/examples/dvx_multi_task.yaml \
  --cfg-options data.spec=/data/lzy/DVisionix/project/data/multicls/dvx_dataset.yaml \
  --work-dir /tmp/p3_multicls --force
# 期望：history.csv 中出现 licence_* 与 logo_* 两组指标；train/val 均跑完
```

## 7. 文件所有权

| 归属 | 文件 |
|---|---|
| Wave A（模型+任务+契约） | `dvisionix/models/classifiers/{multi_task.py(新),__init__.py}`、`dvisionix/models/__init__.py`、`dvisionix/training/tasks/{multi_task_classification.py(新),__init__.py}`、`dvisionix/training/contract.py`、`dvisionix/data/dvx_dataset.py`、`tests/test_models/test_multi_task.py`、`tests/test_training/test_multi_task_task.py`、`tests/test_training/test_contracts.py` |
| Wave B（入口+配置） | `tools/train.py`、`dvisionix/config/schema.py`、`configs/examples/dvx_multi_task.yaml`、`tests/test_training/test_dvx_config_smoke.py` |
| Lead | 本契约、审查、真实数据验收、git 提交 |
| 文档 Wave | `docs/training_contracts.md`、`docs/dataset_spec.md`、`docs/quick_start.md`、`README.md`、`CHANGELOG.md` |
