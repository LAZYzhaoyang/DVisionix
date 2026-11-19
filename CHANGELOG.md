# 变更日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
完整的规划与缺陷清单见 [CodePlan.md](CodePlan.md)。

## [Unreleased]

（暂无 —— 下一版本开发中）

## [2.0.1] — 2026-11-20

**修复与清理**

- `training.early_stopping.restore_best_weights` 现在真正转发给 `EarlyStopping`
  （此前只存在于 schema，配置写了也静默无效，永远用默认 `True`）；新增回归测试。
- 训练结束自动导出（`training.export_best_onnx`）改为复用 P5 导出体系
  （`build_exporter`），并额外落盘 `best.onnx.json` sidecar；返回路径不变。
- 移除已废弃的 `use_giou` 参数（`FCOSDetectionLoss` / `YOLOLoss`），
  统一使用 `giou_weight` / `l1_weight`（v1.1 起已告警，仓库内零使用）。
- 清理未跟踪构建产物（`build/`、`.pytest_cache/`、`__pycache__/`）。

**文档**

- 文档结构重组：历史迁移指南与 v2 实现契约归档到 `docs/history/`，
  用户文档与 CodePlan 同步更新（见同批次文档提交）。

**测试**：1089 passed / 1 skipped / 0 failed。

## [2.0.0] — 2026-11-18

> v2 重构发布：P0 架构基线 / P1 DVX 标准数据集层 / P2 训练接线与三方契约 /
> P3 路由式多任务分类 / P4 推理链路 / P5 可扩展导出体系。
> 接口契约见
> [P0/P1 实现契约](docs/design/v2_p0_p1_contract.md)、
> [P2 实现契约](docs/design/v2_p2_contract.md)、
> [P3 实现契约](docs/design/v2_p3_contract.md)、
> [P4 实现契约](docs/design/v2_p4_contract.md) 与
> [P5 实现契约](docs/design/v2_p5_contract.md)。

**升级提示（主要行为变更，升级前请确认）**

- `ONNXExporter.export()` 返回 `ExportResult`（`str()` / `os.fspath()` 仍返回路径）；
  `verify()` 返回 `VerifyReport`（`__bool__` 保持旧布尔语义）。新增 `tools/export.py`。
- 训练入口执行任务 / 模型 / 数据三级契约校验：任务与模型能力、`data_type`、`num_classes`
  错配会在训练开始前报错；未声明能力的自定义任务跳过对应校验（渐进式契约）。
- `ClsHead` / `MultiLabelHead` 接受 `labels` 参数（修复 `LinearClassifier + cls_head +
  ClassificationTask` 首个 batch 崩溃；margin 头自此在分类任务中真正生效）。
- `strategy: auto` 在 torchrun 下自动识别为 DDP；验证指标聚合改为"先转 CPU 再 all_gather"，
  修复多卡跨设备 `torch.cat` 崩溃。
- DVX 数据集经 `data.dataset: dvx` + `data.spec` 接入训练；`model.num_classes` 必须与
  spec 类别数一致（由契约校验兜底）。`configs/examples/` 提供 cls/det/seg/多任务四份模板。
- 新增推理（`tools/infer.py`、`dvisionix.inference`）与路由式多任务分类
  （`MultiTaskClassifier` / `MultiTaskClassificationTask`）能力。

### 新增：P0 架构基线

- **统一异常族** `dvisionix/errors.py`：`DVisionixError` 及其子类
  （`ConfigError` / `RegistryError` / `DatasetError` / `DatasetSpecError` /
  `AnnotationError` / `SplitError` / `TaskModelMismatchError` / `TaskDataMismatchError` /
  `ModelOutputMismatchError` / `CheckpointError` / `ExportError`），
  `__str__` 自动拼接 `提示：`（hint），且不依赖 torch。
- **分层清单** `docs/architecture/layers.yaml`：R1-R3 的机器可读版本，
  记录 assembly / component / foundation / shared / infrastructure 五层与既有例外。
- **分层检查器** `tools/check_layering.py`：基于 `ast` 的静态检查，支持相对导入绝对化、
  `from`/`to` 与 `from_glob`/`to_glob` 例外匹配、`--json` 输出和 `--no-fail`；
  退出码 0（通过）/ 1（违规）/ 2（清单或目录错误）。
- **门禁测试** `tests/test_architecture/test_layering.py`：真实仓库零违规 +
  R1/R2/R3 自测 + 例外放行 + CLI 退出码。
- `.gitignore` 补充 `.ruff_cache/`、`.mypy_cache/` 工具缓存目录。
- 文档：新增 [架构与分层规则](docs/architecture.md)，说明 R1-R3、清单结构、
  检查器用法、例外登记流程与当前基线。

### 新增：P1 DVX 标准数据集

- **规格与校验** `dvisionix/data/spec/schema.py`：`dvx_dataset.yaml` v1，
  支持 `classification` / `multi_task_classification` / `detection` / `segmentation`
  四种 `data_type`，校验 layout 组合、类别表、`tasks`、`splits`、`parse` 与
  `seg_mode` / `ignore_index`。
- **类别表** `dvisionix/data/spec/classes.py`：`ClassTable`，顺序即 label id，
  拒绝重名，未知标签抛 `AnnotationError`。
- **扫描** `dvisionix/data/spec/scan.py`：`SampleRef` 与 `scan_samples`，
  按 layout 分派到 folder / labelme 扫描器，`rel_path` 相对 `spec.root` 且全局唯一。
- **非破坏性切分** `dvisionix/data/spec/split.py`：`build_splits` 支持
  `ratio`（seed + 分层）/ `dir`（目录映射）/ `list`（JSON/YAML 清单），
  统一做覆盖与不重叠校验；`SplitResult` 可保存/加载。
- **解析器** `dvisionix/data/parsers/`：`folder_per_class` 扫描、labelme
  扫描/解析（rectangle → boxes，polygon → segments）、polygon → mask 栅格化
  （背景默认 0，后写覆盖先写，越界裁剪）。
- **统一数据集** `dvisionix/data/dvx_dataset.py`：`DVXDataset` 按 split 输出
  `Sample` 契约（分类 `label`、多任务 `label`+`task`、检测 `boxes`+`labels`、
  分割按需栅格化 `mask`），并提供 `classes` / `task_names` / `num_classes` /
  `refs` / `split_result` 等只读属性。
- **数据导出** `dvisionix/data/__init__.py` 增加 `DVXDataset` / `DVXDatasetSpec` /
  `SplitResult` / `load_spec`（只加不改）；`Sample.EXTENDED_KEYS` 增加 `task`。
- **工具**：
  - `tools/init_dataset.py`：只读扫描、推断 layout、生成 `dvx_dataset.yaml` 草稿；
  - `tools/check_dataset.py`：数据集只读体检（类别/样本统计、检测框与分割多边形质量、
    `--split-preview` 切分预览、`--json`）。
- **真实数据验收**：`project/data/{cls,det,multicls,seg}/dvx_dataset.yaml`
  （数据在仓库外），对应 `tests/test_data/test_acceptance_real_data.py`：
  2210 / 1646 / 2458 / 766 张，默认切分 1768/442、1317/329、1964/494、613/153。
- **测试**：`tests/test_data/test_dvx_spec.py`、`test_dvx_classes.py`、
  `test_dvx_split.py`、`test_dvx_parsers.py`、`test_dvx_dataset.py`。
- **文档**：新增 [DVX 标准数据集](docs/dataset_spec.md) 与
  [架构与分层规则](docs/architecture.md)；`docs/README.md`、`README.md`、
  `docs/quick_start.md` 同步索引与上手命令；`configs/datasets/` 增加分类 / 检测 /
  分割 / 多任务四份示例模板。

### 新增 / 修复：P2 训练契约、DVX 接线与 DDP

- **三级契约校验** `dvisionix/training/contract.py`（新增）：
  - `check_task_model`：模型能力（`model_task_types`）与 `task.accepted_model_tasks`
    必须有交集；数据集与模型都暴露 `num_classes` 时还必须一致，否则分别抛
    `TaskModelMismatchError` / `TaskDataMismatchError`；
  - `check_task_data`：DVX 数据集的 `data_type` 必须在 `task.accepted_data_types` 内；
    没有 `data_type` 的 `CustomDataset` / 公开数据集预设直接放行；
  - `check_batch_fields`：训练 batch 必须含 `required_batch_fields`，只在第一个训练
    batch 检查一次。
  - 错误消息统一包含「发生了什么 + 提示：如何修复」；未声明能力的自定义任务按
    「渐进式契约」跳过对应校验，内置 7 个任务均已显式声明。
- **模型能力声明** `dvisionix/models/base.py`：新增 `BaseModel.supported_tasks` 与
  `model_task_types()`（优先 `supported_tasks`，回退 `task_type`），并从
  `dvisionix.models` 导出。
- **Task 承担任务知识**：`BaseTask` 增加 `task_type` / `accepted_model_tasks` /
  `accepted_data_types` / `required_batch_fields` 与 `build_transforms()`；7 个内置任务
  （`ClassificationTask` / `MultiLabelTask` / `DetectionTask` / `SegmentationTask` /
  `MaskFormerTask` / `SimCLRTask` / `LinearEvalTask`）逐一声明契约并提供预设变换。
- **训练入口接线**：`Trainer.fit()` 在 `_wrap_model` 之前完成模型 / 数据校验，第一个
  训练 batch 前完成字段校验；失败时不会包装 DDP / 构造优化器，也不会执行任何优化步。
- **DVX 数据集接入** `tools/train.py`：新增 `data.dataset: "dvx"` + `data.spec`
  数据源，按 split 构建 `DVXDataset` 并使用任务的预设变换与数据集自带的
  `collate_fn`；保留 `synthetic` 与公开数据集预设路径。构造顺序改为
  `config → task → transforms(task) → data → model → trainer`，不再按 `task_type`
  硬编码 if/else；任务未提供变换时抛 `ConfigError`。
- **训练配置模板** `configs/examples/dvx_{classification,detection,segmentation}.yaml`：
  `data.spec` 为占位路径的外部数据模板；`tests/test_e2e_configs.py` 将 `examples/`
  排除出 CI 端到端门禁，避免依赖仓库外数据。
- **分类头 labels 兼容**：`ClsHead` / `MultiLabelHead.forward` 接受 `labels=None`。
  此前 `ClassificationTask` 训练步会无条件传 `labels=`，而 `LinearClassifier` 会把
  kwargs 透传给头，最主流的 `LinearClassifier + ClsHead + ClassificationTask` 路径
  在第一个 batch 必崩；现已修复，并新增反射契约测试防止同类问题复发。
- **DDP 修复**：
  - `strategy: auto` 现在同时识别「进程组已初始化且 world_size > 1」与 torchrun 注入的
    `RANK` / `WORLD_SIZE`；torchrun 下日志显示 `strategy: ddp`，入口的
    `Train batches: 8` 经 `DistributedSampler` 切分为每 rank 4 个 batch（`8 → 4/4`）。
  - 多卡验证聚合改为先把 `preds` / `targets` 递归搬到 CPU，再 `all_gather_object`；
    合并后搬回训练设备再更新指标，修复跨设备 `torch.cat` 报
    "Expected all tensors to be on the same device"。支持
    `Tensor` / `dict` / `list` / `tuple`（含检测的
    `(boxes_list, scores_list, labels_list)`）。
  - `tests/test_training/test_ddp_smoke.py` 增加"必须由 torchrun 启动"的 skip 守卫；
    `tests/test_metrics/test_metrics.py` 的 TopK 夹具改为非目标类互不相等，去掉平票
    导致的断言漂移。
- **测试**：新增 `tests/test_training/test_contracts.py`、
  `tests/test_training/test_dvx_config_smoke.py`；全量 `pytest tests -q` 为
  **903 passed / 1 skipped / 0 failed**。
- **真实数据验收**（1 epoch，仅用于确认链路跑通，**不作为模型效果结论**）：
  cls `val_accuracy` 0.9231、det `val_mAP` 0.0176、seg `val_mIoU` 0.1701；
  torchrun 2 卡 DDP 冒烟通过。
- **文档**：新增 [训练契约与自适应训练](docs/training_contracts.md)；
  `docs/dataset_spec.md` / `docs/quick_start.md` / `README.md` / `docs/README.md`
  同步训练接线、契约与示例命令。

### 新增：P3 路由式多任务分类

- **模型** `dvisionix/models/classifiers/multi_task.py`：新增 `MultiTaskClassifier`
  （注册名 `multi_task_classifier`）。共享骨干**只前向一次**，按每个样本的 `task` 用
  `index_select` 取特征子集送进对应分类头；`supported_tasks = ("multi_task_classification",)`；
  只读属性 `task_names`（顺序稳定的 `tuple[str, ...]`）与 `num_classes`
  （`{任务: 类别数}` 的 dict，不是整数）；空 heads / 任务名为空或含 `.` / 头缺
  `num_classes` 均在构建期抛 `ConfigError`。
- **任务** `dvisionix/training/tasks/multi_task_classification.py`：新增
  `MultiTaskClassificationTask`。`required_batch_fields = ("image", "label", "task")`；
  每个任务用自己的样本子集算交叉熵，总损失按 `task_loss_weights` 加权求和；
  `<task>_loss` 记录**未加权** CE、`<task>_acc` 记录该任务 batch 内准确率；
  验证步把 `preds` / `targets` 补齐为模型**全部任务名**的键（未出现任务用空张量占位），
  保证 DDP `all_gather` 键一致；指标按任务惰性创建、跨 epoch 复用，epoch 结束返回
  `<task>_<metric>` 前缀键。
- **契约扩展**：
  - `DVXDataset` 新增只读属性 `task_num_classes`（`{任务: 类别数}`，其他 `data_type`
    返回 `None`；`task=` 过滤样本不影响该表）。
  - `training.contract.check_task_model` 新增**字典类别数**校验：`task_num_classes`
    与模型 `num_classes` 都是字典时，逐任务比较任务名集合与类别数，缺失 / 多余 /
    不一致抛 `TaskDataMismatchError` 并一次列出差异；一边字典一边整数同样报错。
  - 多任务模型配 `ClassificationTask` 由能力校验抛 `TaskModelMismatchError`。
- **训练入口接线**：`dvisionix/config/schema.py::TASK_TYPES` 与
  `tools/train.py::_TASK_MAPPING` 增加 `multi_task_classification`（共 5 个训练任务类型）；
  多任务**不要求**全局 `model.num_classes`，`build_task_cfg` 也**不注入**任务
  `num_classes`（类别数由 heads 自带）。
- **配置模板** `configs/examples/dvx_multi_task.yaml`：`data_type:
  multi_task_classification`、`multi_task_classifier` + `licence` / `logo` 两个 head，
  并显式设置 `training.find_unused_parameters: true`。该目录在 E2E 的
  `NON_TRAINING_DIR_PREFIXES` 中排除，不依赖仓库外数据。
- **DDP**：每个 batch 未必覆盖全部任务，未参与前向的分类头在 DDP 反向时会被判为
  未使用参数，因此多卡训练**必须** `training.find_unused_parameters: true`（单卡无影响）。
  2 卡 torchrun 实测：`strategy: ddp`、每 rank 61/61 train batches、per-task 指标正常
  跨卡聚合。
- **测试**：新增 `tests/test_models/test_multi_task.py`、
  `tests/test_training/test_multi_task_task.py`，扩展 `tests/test_training/test_contracts.py`
  与 `tests/test_training/test_dvx_config_smoke.py`；全量 `pytest tests -q` 为
  **963 passed / 1 skipped / 0 failed**。
- **真实数据验收**（`multicls`，2458 样本，按 `(task, label)` 分层切分 1967 / 491；
  1 epoch，仅用于确认链路跑通，**不作为模型效果结论**）：单卡 12.1s，
  `val_licence_accuracy 0.9581`、`val_logo_accuracy 0.7899`；torchrun 2 卡 DDP 冒烟通过。
- **文档**：新增 [路由式多任务分类](docs/multi_task.md)；`docs/training_contracts.md`
  的任务契约表扩到 8 个并补充字典类别数分支；`docs/dataset_spec.md` /
  `docs/quick_start.md` / `README.md` / `docs/README.md` 同步多任务数据、训练命令、
  特性与文档索引。

### 新增：P4 推理链路

- **预处理规格** `dvisionix/inference/preprocess.py`：新增 `PreprocessSpec`
  （`task_type` / `image_size` / `mean/std/scale`，支持 `from_pipeline` / `to_dict` /
  `from_dict`）、`PreprocessMeta`（`original_size` / `processed_size` / `scale` /
  `image_path`）与 `PreprocessPipeline`。`from_task` 复用 Task 的
  `build_transforms(train=False, image_size=...)`，不重写 resize / normalize；输入支持
  路径、uint8 RGB ndarray、PIL，返回 `(1, 3, H, W)` float32 CPU 张量与反变换元信息；
  同一输入重复调用结果确定，且不修改传入数组。
- **统一结果** `dvisionix/inference/results.py`：`Results` 覆盖分类 `probs` / `topk`、
  检测 `boxes` / `scores` / `labels`、分割 `mask`、`orig_size` / `meta`，检测框为原图
  坐标、mask 为原图尺寸；`to_dict` / `to_json` / `save` / `load` 往返，`visualize`
  返回 BGR 图像（分类 top-k 文本、检测框 + 分数、分割调色板叠色 0.5）。
- **三个 Predictor** `dvisionix/inference/predictors.py`：`BasePredictor` 负责设备解析、
  预处理、`eval()` + `no_grad()` 前向与训练状态恢复（`finally` 中恢复）；
  `ClassificationPredictor` 做 softmax + top-k；`DetectionPredictor` 复用
  `model.decode()`，按 `meta["scale"]` 把框反变换回原图并 clip；`SegmentationPredictor`
  先 argmax 再 nearest 插值回原图，可选 `return_probs`；`build_predictor` 按任务类型
  返回子类。`from_checkpoint` 兼容纯 state_dict / `model_state_dict` / DDP / compile
  前缀，默认 `weights_only=True`，完整 Trainer checkpoint 需 `trusted=True`。
- **CLI** `tools/infer.py`：`--config` / `--checkpoint` / `--input` / `--output` 必填，
  支持 `--task-type` / `--device` / `--trusted` / `--save-json` / `--no-save-json` /
  `--save-vis` / `--cfg-options`；目录递归收集并按路径排序，逐图输出 `<stem>.json`
  （`--save-vis` 时另有 `<stem>_vis.jpg`），stdout 打印汇总，失败非零退出。
- **错误** `dvisionix/errors.py` 新增 `InferenceError`；`dvisionix/__init__.py`
  懒加载子模块与 `__all__` 增加 `inference`。
- **测试**：新增 `tests/test_inference/`（preprocess / results / predictors / CLI）；
  全量 `pytest tests -q` 为 **1066 passed / 1 skipped / 0 failed**。
- **真实数据验收**（1 epoch checkpoint，仅用于确认链路跑通，**不作为模型效果结论**）：
  cls 原图 870×870、5 类概率和为 1.0、top-1 类别为 2 且概率 0.903；det 原图 439×439、
  检出 4 个框且全部落在原图界内；seg 原图 1024×1024、mask 为 `(1024, 1024) int64`、
  取值样例 `[0, 3, 5, 10]`。反例：不加 `--trusted` 加载完整 checkpoint 会抛
  `CheckpointError` 并提示需要 `trusted=True`。
- **文档**：新增 [推理链路](docs/inference.md)；`docs/quick_start.md` / `README.md` /
  `docs/README.md` 同步推理命令、特性、测试基线与文档索引。

### 新增：P5 导出接口与 ONNX 一致性校验

- **导出契约** `dvisionix/export/base.py`（新增）：`EXPORTERS` 注册表（复用
  `Registry`）；`ExportResult`（`path` / `format` / `input_names` / `input_shapes` /
  `output_names` / `opset` / `dynamic_axes` / `preprocess` / `parity_max_abs_diff` /
  `extra`，支持 `to_dict` / `from_dict` / `save` / `load` 与 JSON sidecar）；
  `VerifyReport`（`passed` / `max_abs_diff` / `outputs`，`__bool__` 等价 `passed`）；
  `BaseExporter` 抽象基类；`build_exporter(format, model, **kwargs)` 按注册名构建，
  未注册格式抛 `ConfigError` 并列出全部已注册格式。
- **ONNXExporter 重构** `dvisionix/export/onnx_exporter.py`：继承 `BaseExporter`、
  注册名 `onnx`；`export()` 返回 `ExportResult`（`input_shapes` 为 dummy 完整形状，
  `dynamic_axes` 为实际生效映射，`extra` 含 `backend` / `simplify` /
  `output_path_map`）；新增可选构造参数 `preprocess`，缺省时按 `task_type` +
  `normalize` 与输入形状推断 `PreprocessSpec`（缺省 ImageNet，`scale=1/255`）；
  `verify()` 返回 `VerifyReport`，汇总全局 `max_abs_diff` 与逐输出明细，并回填
  `result.parity_max_abs_diff`；ONNX `metadata_props` 新增 `dvisionix_export` 键
  （完整 `ExportResult.to_dict()`），保留原有 `normalize_*` 与 `output_path_map`。
- **TensorRT 扩展点** `dvisionix/export/tensorrt.py`（新增）：`TensorRTExporter`
  注册名 `tensorrt`，可构建；`export()` 恒抛 `NotImplementedError`（明确未实现，
  不假装支持）。
- **CLI** `tools/export.py`（新增）：`--config` / `--checkpoint` / `--output` 必填，
  支持 `--format` / `--dynamic-batch` / `--opset` / `--verify` / `--trusted` /
  `--device` / `--cfg-options`；导出后落盘 `<output>.json` sidecar（**先校验、后落盘**，
  保证 `parity_max_abs_diff` 已回填）；`--verify` 未通过以退出码 1 结束；`--trusted`
  与 `tools/infer.py` 语义一致。
- **向后兼容（返回类型有变化，请留意）**：
  - `ONNXExporter.export(...)` 由返回 `str` 改为 `ExportResult`；该类实现 `__str__`
    与 `__fspath__`，`str(result)` / `os.fspath(result)` / `open(result)` /
    `os.path.exists(result)` 仍可用；但 `result == "path"` 不再成立。
  - `ONNXExporter.verify(...)` 由返回 `bool` 改为 `VerifyReport`；`__bool__` 返回
    `passed`，既有 `assert exporter.verify(path)` 语义不变；`is True` / `== True`
    不再成立。
  - 构造函数签名保持兼容，仅**新增可选** `preprocess`；`from dvisionix.export import
    ONNXExporter` 不变。
- **导出面** `dvisionix/export/__init__.py`：完整导出 `ONNXExporter` / `BaseExporter` /
  `ExportResult` / `VerifyReport` / `EXPORTERS` / `build_exporter` /
  `TensorRTExporter`。
- **测试**：新增 `tests/test_export/test_export_base.py`（结果序列化 / 路径协议 /
  注册表 / TensorRT 占位 / metadata / verify 回填）与
  `tests/test_export/test_export_cli.py`（CLI 端到端 / `--verify` / `--trusted`）；
  既有 `test_onnx_export.py` / `test_onnx_contract.py` 原样通过；全量
  `pytest tests -q` 为 **1092 passed / 1 skipped / 0 failed**。
- **真实 checkpoint 验收**（1 epoch 冒烟 checkpoint，仅链路证据，**不作为效果结论**）：
  分类 `[1,3,224,224]` → `[1,5]`，`max_abs_diff` 5.96e-06；检测 `[1,3,512,512]` →
  8 个输出（`cls_0..3` / `reg_0..3`），3.34e-06；分割 `[1,3,256,256]` →
  `[1,18,256,256]`，4.77e-06；三者均 < 1e-4。sidecar 的 `preprocess` /
  `input_shapes` 正确，`--verify` 时 `parity_max_abs_diff` 已回填（同一分类
  checkpoint 复跑 2.19e-05，属正常波动，量级约 1e-5）。
- **文档**：重写 [模型导出](docs/model_export.md)（导出体系 / `ExportResult` 字段 /
  CLI 参数与示例 / 向后兼容 / ONNX metadata / ORT 校验 / TensorRT 扩展点 / 已知限制）；
  `docs/quick_start.md` 新增「导出为 ONNX（新 CLI）」；`README.md` /
  `docs/README.md` 同步特性、命令、测试基线与文档索引。

### 尚未接线 / 已知边界

- `DVXDataset` 仍未注册到 `DATASETS`；训练入口通过 `data.dataset: dvx` + `data.spec`
  的专用分支构建，`build_dataset({...})` 不能按名字直接构建。
- 多任务分类（P3）只支持路由式分类：margin 头（ArcFace/CosFace 等）需要 `labels`，
  而路由前向不向 head 透传 `labels`，暂不支持（强行配置不会报错，但 margin 会静默失效）；
  `synthetic` 数据源不支持多任务；跨任务共享标签空间未实现。
- DDP 已验证范围为单机多卡；多机（多节点）DDP 未验证。
- P2 范围不含推理与导出链路（P4 已补单任务推理）；P3 也未做多任务推理/导出专项。
- P4 只接入单任务推理（classification / detection / segmentation）；P5 已交付 ONNX
  导出与一致性校验，但多任务分类推理、ONNX / TensorRT **运行时推理**与批量吞吐优化
  仍未实现。
- TensorRT 导出只是扩展点：`TensorRTExporter.export()` 恒抛 `NotImplementedError`；
  未做量化 / 裁剪 / FP16 导出，quantization 与引擎构建不在 P5 范围。
- `seg_mode: query` 只是 schema 保留值，数据层仅实现 `semantic`。
- 检索（retrieval / re-identification）任务尚未实现。
- `ignore_index` 目前只做 spec 校验与展示，`DVXDataset` 栅格化背景固定为 0；
  训练配置的 `ignore_index` 需与规格保持一致。

## [1.1.1] — 正确性与稳定性修复

> ⚠️ **本批修复会改变检测指标与部分训练结果的口径**，历史实验的 `val_mAP` /
> `val_mAP_50` / `val_mAP_75` 与修复后**不可直接比较**，详见下方「会改变结果的变更」。

### 修复：检测解码的索引错位（严重）

`yolo_decode` / `fcos_decode` / `retinanet_decode` 把分数展平后的索引含义搞反了：
`scores` 由 `(N, C)` 行主序展平，索引是 `idx = loc * C + cls`，因此**类别应取模、
位置应整除**；三处却都拿"位置数"当除数。

后果是**标签与框同时错位**，例如受控输入下应得 `label=1、中心 (3,3)`，
实际得到 `label=2、中心 (1,3)`。训练 loss 完全正常（损失端语义正确），
因此从单元测试到端到端门禁都没有任何信号——原有断言只检查
`boxes.shape[1] == 4` 与三个列表长度相等。

影响范围：所有检测器的**验证指标、推理输出与 ONNX 导出结果**。
新增 `tests/test_models/test_decode_correctness.py` 用「已知类别 + 已知位置」的
受控输入把 `(label, box)` 对应关系钉死。

### 修复：AMP(fp16) 下的数值失稳

| 位置 | 问题 |
|---|---|
| `losses/detection/box_loss.py` | fp16 下面积和溢出 65504 → `union=inf` → **GIoU loss = -inf**（边长 ≈181px 即触发）；闭合框面积 `cw*ch` 同样溢出 → `giou=nan` |
| `losses/detection/losses.py`（CenterNet 热图 focal） | fp16 下 `1-1e-4` 舍入成 `1.0`，clamp 上界失效 → `log(0)` → **loss = nan** |
| `losses/segmentation.py`（MaskFormer/Mask2Former） | `F.binary_cross_entropy` 对 autocast 不安全 → **直接抛 RuntimeError**；改用 `binary_cross_entropy_with_logits` |
| `heads/classification/{arcface,sphereface,adaface,curricularface}.py` | fp16 下 `1.0-1e-7` 舍入成 `1.0` → `acos` 反传 -inf → **梯度全 NaN** |

此前的 CI 只装 CPU-only torch，`torch.autocast` 恒为关闭，这些路径从未被执行。

### 修复：其他静默错算

- **`DiceLoss` / `CombinedSegmentationLoss` 遇 `ignore_index=255` 直接崩溃**：
  255 被原样送进 `F.one_hot` → `Class values must be smaller than num_classes`。
  掩膜含 void（Cityscapes / ADE20K）即不可用。
- **`LinearClassifier` 丢弃 `**kwargs`**，且 `ClassificationTask` 只传 `images`，
  导致 ArcFace / CosFace / SphereFace / AdaFace / CurricularFace 五个 margin 头
  **从未施加过 margin**（静默退化成 NormFace）。
- **`RetinaNetLoss` 缺 batch 归一化**：loss 与 batch size 严格成正比
  （实测 bs=1/2/4 → 0.971 / 1.942 / 3.884），等效学习率随 batch 漂移。
- **`SegFormerV3Head` 的通道注意力被平方**：`fused * self.se(fused)` 实为
  `fused²·scale`（`SEBlock.forward` 已返回 `x*scale`），符号信息被抹掉。
- **`UPerNetHead.fpn_convs` 是死参数**（定义后从未使用）：永远拿不到梯度，
  DDP 下 `find_unused_parameters=False` 时第二轮就会中断训练。
- **`ConvNormAct(norm="ln")` 归一化错误的轴**：在 `(B,C,H,W)` 上建 `nn.LayerNorm(C)`
  归一化的是最后一维 `W`；`W != C` 直接报错，`W == C` 静默算错。改用 `LayerNorm2d`。
- **8 个度量头用 legacy `torch.FloatTensor`** 建权重，无视 `torch.set_default_dtype`
  与 `set_default_device`；`float64` 下直接 matmul dtype 冲突。改用 `torch.empty`。
- **`DINOLoss` 去噪分支把归一化 `cxcywh` 当 `(x1,y1,w,h)`** 转像素，框位置随自身
  尺寸偏移、GIoU 失去平移不变性；同时 `dn_loss` 的 GIoU 项漏乘 `giou_weight`。

### 新增

- `tests/test_models/test_decode_correctness.py`：解码的 (label, box) 对应关系契约。
- `tests/test_models/test_amp_numerics.py`：fp16 / autocast 数值稳定性、
  margin 透传、batch 归一化、通道注意力次数、LayerNorm 轴、死参数等回归。
- `tests/test_training/test_gpu_availability.py`：CUDA 上的断点续训与线性评估。
- `tests/test_training/test_reliability_fixes.py`：非标量指标、resume 语义、
  早停恢复、checkpoint 前缀、超参搜索指标读取。
- `tests/test_training/test_ddp_correctness.py`：设备绑定、rank0 落盘、
  早停同步、工作目录确定性、空验证集检出。

### 修复：GPU 可用性

- **GPU 上断点续训必然崩溃**：`Trainer.load_checkpoint` 用
  `map_location=self.device`，device 为 cuda 时整份 checkpoint（含
  `rng_state["torch"]` 这个 **CPU** ByteTensor）被搬到 GPU，随后
  `torch.set_rng_state()` 抛 `TypeError: RNG state must be a torch.ByteTensor`。
  改为一律先加载到 CPU。CI 装的是 CPU-only torch，`map_location="cpu"` 是空操作，
  因此该缺陷在 CI 中永远不会暴露。
- **`LinearEvalTask` 在 GPU 上必然崩溃**：线性头以默认设备（CPU）创建且从未搬移，
  而特征来自已搬到 CUDA 的 backbone → `Expected all tensors to be on the same
  device`。官方 `configs/classification/linear_eval.yaml` 的 `device: auto` 在
  有 GPU 的机器上就是 cuda，即该配置在 GPU 上从来跑不起来。改为按模型参数的
  device/dtype 建头。
- **线性头不进 checkpoint**：线性头是 Task 持有的参数，不在 `model.state_dict()` 内，
  而此前没有任何 Task 实现 `state_dict()`（`Trainer` 早已支持该契约）。
  续训会重新随机初始化线性头却仍加载旧头的 Adam 动量。已为 `LinearEvalTask` 实现
  `state_dict()/load_state_dict()`。

### 修复：训练可靠性与工具链

- **非标量指标直接崩训练**：`per_class=True` / `average="none"` 返回 `list`，
  epoch 汇总处 `float(list)` 抛 `TypeError`，训练在第一个 epoch 末尾崩溃。
  现展开为 `key_0/key_1/...`（保留逐类信息）。
- **`resume: true` 被当成 checkpoint 路径**：`torch.load(True)` 抛
  `AttributeError: 'bool' object has no attribute 'seek'`，且会另起一个带新时间戳的
  目录。现 `true` 等价于 `"auto"`。
- **续训后 `restore_best_weights` 静默失效**：`EarlyStopping.best_weights` 是内存副本、
  不进 checkpoint，续训后恒为 `None`，触发早停时既不恢复也不提示。现改为从
  `checkpoints/best.pt` 兜底恢复，无法恢复时明确告警。
- **checkpoint 存的是 DDP / torch.compile 包装后的键**（`module.` / `_orig_mod.`）：
  跨模式续训在 `strict=True` 下抛 "Missing/Unexpected keys"（元信息校验因 unwrap 过
  反而通过，误导性极强）；`load_backbone` 的前缀匹配全部落空并回退成整份 state_dict，
  结果是**一个权重都没加载**却只留一条 warning。现保存统一存未包装的键，
  加载时按目标模型自动剥离/补回前缀。
- **超参搜索的监控指标列恒为空**：`--work-dir` 是工作根，`resolve_work_dir` 会在其下
  再拼 `<experiment>/<时间戳>-<哈希>`，而汇总读的是 `work/history.csv` → 永远读不到；
  `subprocess.run(check=False)` 还会吞掉 trial 失败。现按有限深度定位 `history.csv`
  并记录非零退出码。

### 修复：DDP 正确性（CI 无多卡，这些分支从未被执行）

- **`--devices` 是死参数**：`self.devices` 只被赋值、全包再无读取，DDP 设备完全由
  `LOCAL_RANK` 决定 —— 用户以为用了 1、3 号卡，实际仍用 0、1 号卡且无提示。
- **显式 `device: cpu` 被静默改成 cuda**，且 NCCL/Gloo 后端按 `cuda.is_available()`
  而非实际设备选择（会拿 NCCL 去 all-reduce CPU 张量）。
- **`history.csv` / `best_metrics.csv` / JSONL / TensorBoard 被所有 rank 并发写**，
  而 rank≠0 的 `val_*` 恒为 0，可能写出整列 0 的历史。现仅 rank0 落盘。
- **DDP + 早停会死锁**：各 rank 用各自指标决策（rank≠0 看到 0），rank0 退出 epoch
  循环后其余 rank 继续进入下一轮集合通信 → 永久 hang；而早停默认开启。
  现早停只在 rank0 决策，epoch 末尾广播给所有 rank。
- **torchrun 下各 rank 的工作目录不同**：目录名带 `datetime.now()` 时间戳，跨秒启动
  就会分家。现分布式下用确定性目录名（配置哈希，无时间戳），无需通信。
- **验证集被强制 `drop_last=True` 可能被清空**：此时验证循环一次都不执行、
  `val_loss` 不存在、best.pt 永不产出、指标恒为 0，却完全不报错。现显式报错。

### 修复：测试与文档

- `tests/test_training/test_ddp_smoke.py` 断言的 `"accuracy"` 键自 v1.1 起已改名
  `val_accuracy`（见 1.2 节），该 GPU-only 用例一旦真的运行就会失败，现已同步。
- `ONNXExporter.export(backend="dynamo")` 与默认 `dynamic_batch=True` 自相矛盾：
  最自然的默认调用必然抛 `ValueError`。现用 `None` 哨兵区分"未指定"与"显式要求"，
  默认调用可用，显式要求动态维度时仍明确报错（原有契约测试继续通过）。
- 移除从未被代码引用的核心依赖 `matplotlib`（全仓库仅出现在 pyproject / 文档里）。
- 修复 `dvisionix/models/layers/timm_layers.py` 中机械损坏的 docstring。
- `configs/classification/linear_eval.yaml` 此前被端到端门禁当作"模板配置"排除，
  现补上一条真正训练它的用例（`device=auto`，CI 上走 CPU 也会执行）。

### 会改变结果的变更

- **检测指标口径**：解码索引修正后 `val_mAP` / `val_mAP_50` / `val_mAP_75`
  以及所有推理/导出结果都会变化。**修复前的检测指标不可信**，历史实验请勿直接对比。
- **度量学习头**：margin 现在真正生效，使用这些头的训练结果会明显不同。
- **RetinaNet 训练**：loss 量级下降为原来的 1/batch_size 量级，等效学习率提高，
  既有超参（尤其是 lr）需要重新标定。
- **SegFormerV3 / UPerNet / DINO**：对应分支的数值与参数参与情况发生变化。

## [1.1.0] — 稳定性与工程优化

阶段 0-5 全部完成：装配层门禁 → 修复跑不通的官方配置 → P0 正确性 →
训练/评估/导出闭环 → 打包与 CI → 性能优化。
20 项已核实缺陷 D1-D20 全部修复并附回归测试；官方配置由 14/17 可跑提升到 **17/17**。

> 会改变训练结果、输出格式或 API 行为的变更，见
> [docs/v1.1_changes.md](docs/v1.1_changes.md)（迁移指南）。

### 新增

- **装配层门禁**：`tests/test_e2e_configs.py` 对 `configs/` 下每个可训练配置跑 1 个 epoch
  （此前所有「配置加载」测试都止步于 `build_model()`，导致 3 个官方配置在 286 条测试全绿时无法训练）。
- **模型契约门禁**：`tests/test_model_contracts.py` 断言所有检测器的 `decode()`
  接受统一关键字参数。
- **CPU gloo 双进程 DDP 一致性测试**：`tests/test_ddp_cpu.py`，无需 GPU 即可在 CI 长期运行。
- **ONNX 导出契约测试**、**checkpoint 契约测试**、**统一 Evaluator 契约测试**、
  **损失语义与裁剪契约测试**。
- `training.ema.*` 配置项（EMA 此前只有编程式入口）。
- LICENSE、CHANGELOG、Dependabot、发布工作流。

### 修复

| 编号 | 问题 |
|---|---|
| D1 | `centernet` / `nmsfree_yolo` 的 `decode()` 缺少 `iou_threshold`，验证阶段必然抛 `TypeError` |
| D2 | `yolov10_synthetic.yaml` 向 `OneToOneYOLOLoss` 传它不接受的 `topk` |
| D3 | `simclr` 不在合法任务类型中；且合成数据缺少 simclr 分支 |
| D4 | DINO 用 `feats[0].shape * 4` 当图像尺寸，官方配置下偏差 2 倍，去噪分支与主分支坐标系不一致 |
| D5 | DDP 聚合把检测的 `(boxes, scores, labels)` 3-tuple 拍平成 6-tuple |
| D6 | FCOS / YOLO 回归损失重复累加 L1（`use_giou=True` 实为 GIoU+L1，`False` 实为 2×L1） |
| D7 | 检测损失未按正样本归一化，量级随 batch size 与层数漂移 |
| D8 | `Trainer.validate()` 无 DDP 分支且不触发验证回调（EMA 权重交换失效） |
| D9 | 检测管线先 resize 到目标尺寸再同尺寸裁剪，随机裁剪完全失效 |
| D10 | `ToTensor` 可能把 mask 转成 float，破坏 `CrossEntropyLoss` 的 long 契约 |
| D11 | 首图空预测时 mask AP 的目标尺寸退化为 `(1, 1)` |
| D12 | ONNX 导出不支持嵌套输出；导出会原地修改调用者模型的设备与 train/eval 状态 |
| D13 | 合成数据 train/val 共用文件名序列，验证集是训练集的子集 |
| D14 | `dvisionix/config/defaults/*.yaml` 不进 wheel，非 editable 安装下 `Config.from_default()` 必然失败 |
| D15 | 依赖与版本号在 `setup.py` / `requirements.txt` 双源且已漂移 |
| D16 | CI 无 mypy / coverage / 安全检查，仅测 3.10 与 3.11 |
| D17 | EMA / `DistillCallback` 在配置驱动路径下不可达 |
| D18 | 任务类型常量存在三份且不一致（其中一份是从未被引用的死常量） |
| D19 | 3 处 docstring 机械损坏 |
| D20 | 测试只「构建模型」不「运行配置」，配置是否可用无人验证 |

另修复：梯度累积尾窗分母错误、`fit()` 重复调用 `on_validation_epoch_end()` 导致
history 中出现基于空累加器算出的指标列、两个「loss 下降」测试的顺序相关偶发失败。

### 变更（可能影响历史结果可比性）

- **损失语义**：FCOS / YOLO 默认改为 **GIoU only**（`giou_weight=1.0, l1_weight=0.0`）；
  旧的 `use_giou` 开关按字面意图迁移并发出 `DeprecationWarning`。
  v1.0.0 的检查点不可直接续训，既有超参需重新标定。
- **指标口径**：检测损失各分量改为真实均值；`val_loss` 与 v1.0.0 不可直接比较。
- **输出格式**：`history.csv` 的 epoch 级指标统一为 `val_` 前缀
  （`accuracy` → `val_accuracy`，与 `val_loss` 一致）。
- **增强生效**：检测训练管线改为先放大到 1.1× 再随机裁剪（此前增强完全未生效）。
- **裁剪契约**：`RandomCrop` / `CenterCrop` / `BoxSyncRandomCrop` 在输入过小时默认报错
  （`on_small="error"`），不再静默返回错误尺寸。
- **数据口径**：合成数据 train / val 使用不同 split 与独立随机种子（消除泄漏），
  且数据变为可复现。
- **resume 校验**：`load_checkpoint` 默认拒绝配置哈希 / 任务类型 / 模型类型不匹配的
  checkpoint，并拒绝纯 `state_dict`；可用 `allow_config_mismatch=True` 显式放行。
- **打包**：`setup.py` 已删除，元数据统一到 `pyproject.toml`。

## [1.0.0]

功能基线。配置驱动 + 组件化模型库 + 统一训练引擎 + 工具链，API 冻结进入稳定期。

- 12 种骨干、3 种颈部、35 个头、12 个检测器、25+ 损失
- 统一 Trainer：DDP、AMP、梯度累积、完整 resume、torch.compile、channels_last
- 分类 / 检测 / 语义分割 / 实例与全景分割 / 自监督对比学习 + 线性评估
- 286 条测试、`ruff` / `black` 全绿、10 篇专题文档

## [0.17.0]

训练工程 P2（超参搜索 / 特征蒸馏）+ P3（性能开关 / 实验管理）+ DINO look-forward-twice。

## [0.16.0]

DINO-lite、线性评估闭环、训练工程 P1。

## [0.15.0]

组合器目录化、model 分层重构、SwinV2 / DeformableV2 / SegFormerV3。

## [0.14.0]

中期模型扩充：ConvNeXtV2 / EfficientNetLite / MiT 骨干 + SwinUNet + YOLOv11。

## [0.13.0]

model 分层重构 + layers 统一 + 调用规则 R1-R6 入册。

## [0.12.0]

ViT / Swin 骨干、YOLOv9-lite(PGI)。

## [0.11.0]

内置骨干体系、SimCLR 端到端、分割增强。

## [0.10.0]

YOLOv7 / YOLOv10、CenterNet、BiSeNet、Circle / SimCLR 分类头。

## [0.9.0]

全景评估、RT-DETR-full、Mask2Former 完整版。

## [0.8.0]

PSP / UPerNet / DeepLabV3+、DeformableDETR、度量学习头。

## [0.7.x]

decode 归位到模型文件、组合性验证。

## [0.6.0]

Mask2Former 完整版、EMA / 蒸馏回调、CI。

## [0.5.0]

YOLO / DETR、SegFormer / MaskFormer、度量学习。

## [0.4.0]

模型模块丰富（FCOS / RetinaNet / UNet / DeepLab 等）。

## [0.3.0]

训练子系统重构：Task 组件化、loss 迁移到模型层、DDP / resume / work_dir。

## [0.2.0]

组件化重构：Registry + 配置驱动入口。
