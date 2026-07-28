# DVisionix 重构计划（Refactor Plan v0.2）

> 本文档取代旧的开发进度清单，作为算法库重构的唯一执行依据。
> 目标：把当前“教学脚手架”升级为 **配置驱动 + 组件注册** 的标准视觉算法库。

---

## 一、重构目标

1. **配置驱动**：所有可替换组件（model / data / task / loss / metric）均由 YAML 配置构建。
2. **组件注册**：统一 Registry 机制，新增组件“注册即可用”，不改 `__init__` 硬编码。
3. **模型组件化**：`backbone → neck → head` 三段式，后处理与模型解耦。
4. **训练引擎现代化**：AMP、梯度累积、完整 resume、确定性、按样本加权指标。
5. **正确性优先**：检测 mAP / NMS 对接成熟实现，去掉“简化版”自研逻辑。
6. **代码卫生**：修复全部乱码、依赖冲突、打包 bug，兑现或删除文档承诺。

---

## 二、目标目录结构

```
dvisionix/
  registry.py          # 新增：统一 Registry + build_from_cfg
  config/              # 保留 Config，补 schema 校验 + CLI override
  data/
    backends/          # torchvision/COCO/VOC 真实数据源封装
    adapters/ datasets/ transforms/ collate.py
  models/
    base.py            # 强化 BaseModel 契约 + task_type 枚举
    backbones/         # timm features_only 多尺度 + out_channels
    necks/             # FPN 等（新增）
    heads/             # ClsHead/SegHead/DetHead（新增）
    detectors/         # 组装器，含 decode 包装（后处理独立）
    toy/               # SimpleCNN 等玩具网络，仅测试/教学
    postprocess.py     # NMS/decode 集中于此
  training/            # Trainer/Task/Callback/losses（去调度器双轨）
  metrics/             # 对接 torchmetrics/pycocotools
  utils/  export/
tools/
  train.py             # 统一 config 驱动训练入口
```

---

## 三、已识别问题清单（重构依据）

### 代码卫生（P0）
- 多文件中文注释乱码（GBK↔UTF-8 混淆）：models/detection.py、models/postprocess.py、
  config/config.py、data/factory.py、data/collate.py、data/transforms/third_party.py、
  metrics/base.py、utils/device.py、utils/logger.py、export/*、setup.py、requirements.txt 等。
- setup.py 使用 os.path.exists 但未 import os，安装会报错。
- 依赖冲突：setup.py 写 torch>=2.12.0（不存在），requirements.txt 写 torch>=2.0.0。
- README 宣称“覆盖率>80% / black / flake8 / mypy”未兑现，需补齐或删除。

### 架构（P1）
- 无注册表：models/tasks/transforms 硬编码 import，DatasetFactory 靠字符串反射，脆弱。
- Config 半配线：train_from_config.py 只认分类，检测/分割配置无法真正驱动。
- 调度器双轨：Trainer.fit 内部 scheduler.step() 与 LearningRateScheduler 回调重复，易 double-step。
- Task 把优化器/调度器写死（lr、T_max 不可配）。

### Model 层（P1，重点）
- 三套互不兼容体系并存：SimpleCNN/SimpleSegmentationModel/SimpleDetectionModel、TimmBackbone/TimmClassifier、GridDetectionModel。
- 两个检测模型互相矛盾：SimpleDetectionModel 输出 dict 但无 loss/decode/task（死代码）；只有 GridDetectionModel 接入训练链路。
- 输出契约不统一：有的返回 Tensor，有的返回 dict，decode/NMS 塞进模型内部。
- BaseModel 契约过弱，task_type 只是随手字符串。
- TimmBackbone 用 num_classes=0 只取全局向量，丢弃多尺度特征，无法喂检测/分割。
- 无 backbone/neck/head 分层，无 from_config，无预训练权重管理。

### Trainer 能力缺口（P1）
- 无 AMP、无梯度累积、无 DDP。
- resume 不完整：load_checkpoint 未在 fit 调用；EarlyStopping/scheduler/scaler 状态未存。
- 指标聚合按 batch 均值再平均，最后不满 batch 失真，应按样本数加权。
- on_batch_end 被 log_interval 截断，导致 TensorBoardLogger 丢数据。
- 到处 print，未使用现成的 utils.logger；无 seed 设置。

### 数据管线（P2）
- BaseDataset._standardize_format 在 __getitem__ 硬编码 ImageNet 归一化，与 transforms 的 Normalize 重复。
- DetectionResize 不裁剪越界框；AlbumentationsWrapper 不处理 bbox；MetricCollection 检测分支为 pass。

### Metrics / 后处理（P2）
- DetectionMetrics 自研简化实现，compute 与 _compute_map_at_iou 重复计算，正确性/效率存疑。
- GIoULoss 假设框已配对，缺正负样本匹配。

---

## 四、分阶段执行计划

### 阶段 0 — 卫生清理（P0）
- [ ] 修复全部乱码文件（统一 UTF-8 重写注释/docstring）。
- [ ] 修 setup.py（import os），统一 torch 版本约束（torch>=2.0,<3）。
- [ ] 对齐 README/文档承诺（补工具链配置或删除未兑现说法）。
- [ ] 引入 pyproject.toml + black/ruff/mypy 配置。

### 阶段 1 — Registry 与 Config 配线（P1）
- [ ] 新增 dvisionix/registry.py：Registry + build_from_cfg。
- [ ] 定义 MODELS/BACKBONES/NECKS/HEADS/DATASETS/TRANSFORMS/TASKS/LOSSES/METRICS 注册表。
- [ ] 组件用装饰器注册，去掉 __init__ 硬编码依赖。
- [ ] Config 增加 schema 校验 + CLI 覆盖（--cfg-options a.b=c）。
- [ ] 新增 tools/train.py 统一入口，config 真正驱动 data+model+task+trainer。

### 阶段 2 — Model 层重构（P1）
- [ ] 强化 BaseModel 契约：task_type 枚举、forward 只出 raw 预测、from_config、init_weights。
- [ ] TimmBackbone 改 features_only=True + out_indices，输出多尺度 + out_channels。
- [ ] 新增 necks/（FPN）与 heads/（ClsHead/SegHead/DetHead）。
- [ ] 新增 detectors/ 组装器；decode/NMS 移至 postprocess.py。
- [ ] 玩具网络移入 models/toy/，标注“仅测试”；删除死代码 SimpleDetectionModel。
- [ ] GridDetectionModel 标注 demo 或重写为 backbone+DetHead 可用版本。

### 阶段 3 — Trainer 能力补强（P1）
- [x] 去调度器双轨（保留回调，移除 fit 内部 step）。
- [x] 新增 AMP（autocast+GradScaler）、梯度累积、seed/确定性。
- [x] 指标聚合改为按样本数加权。
- [x] on_batch_end 脱钩 log_interval，打印频率交回调自身。
- [x] 完整 resume：fit 调用 load_checkpoint；保存 optimizer/scheduler/scaler/callback 状态。
- [x] 统一日志走 utils.logger，去掉散落 print。
- [ ] （可选）DDP/多卡（暂缓，后续按需求引入）。

### 阶段 4 — 数据管线澄清（P2）
- [x] 归一化职责唯一化：BaseDataset 感知 transforms.provides_normalization；transforms 内 Normalize 默认启用，双重归一化路径打 DeprecationWarning。
- [x] DetectionResize 补越界裁剪 + 退化框过滤；AlbumentationsWrapper 自动补 bbox_params 并回写 boxes/labels。
- [ ] （后续）真实数据源迁入 data/backends/，DatasetFactory 走全局 DATASETS 注册表。

### 阶段 5 — Metrics / 后处理正确性（P2）
- [x] 检测 mAP 支持 torchmetrics 可选后端（use_torchmetrics=True），无依赖时回退内置实现。
- [x] MetricCollection 检测分支从空 pass 补为标准派发（outputs.pred_* + batch.boxes/labels）。
- [x] GIoULoss 文档明确用途：正负样本匹配由 task 侧负责，Loss 仅接收匹配后的 (N,4) 对。
- [ ] （后续）分类/分割指标全面迁移到 torchmetrics（当前内置实现在小规模数据下可用）。

### 阶段 6 — 文档 / 测试 / 示例收尾（P2）
- [x] 补齐单测：registry 构建、model 前向 shape、trainer 一步训练、resume、AMP、指标数值（68 passed）。
- [x] demos 顶部统一加迁移提示指向 tools/train.py；verify_*.py 加提示指向 pytest。
- [x] 更新 README：架构概览、config 驱动用法、0.1.x→0.2.0 迁移说明。

---

## 五、执行顺序与依赖

- 必须先做：阶段 0 → 阶段 1（地基）。
- 可并行：阶段 2（Model）与阶段 3（Trainer）。
- 依赖注册表：阶段 2/4/5。
- 建议节奏：0 → 1 → (2 ∥ 3) → 4 → 5 → 6。

## 六、兼容与风险

- 破坏性变更：model 命名空间、Trainer 调度行为、数据归一化位置将 break 现有 demo。
- 计划在 0.2.0 打破，保留迁移说明；玩具网络保留在 toy/ 以免破坏教学示例。
- 检测第三方依赖通过 extras_require 可选安装。

---

## 七、进度记录

- [x] 阶段 0：卫生清理
  - 丢弃“乱码修复”任务：经验证，所谦“乱码”文件在磁盘上均为合法 UTF-8，仅为 PowerShell GBK 代码页显示假象。
  - 修复 setup.py（import os）、统一 torch 版本约束（>=2.0,<3）、新增 pyproject.toml（black/ruff/mypy/pytest）、版本号 0.2.0、README 软化未兑现承诺。
  - 基线测试 38 passed。
- [x] 阶段 1：Registry + Config 配线
  - 新增 dvisionix/registry.py（Registry + build_from_cfg + 9 个全局注册表）。
  - 注册 models/tasks/losses/metrics/datasets 组件 + 小写别名，提供 build_model/build_task/build_loss/build_metric/build_dataset 并在顶层导出。
  - Config 新增 update_from_cli / parse_cli_options（--cfg-options a.b=v）。
  - 新增 tools/train.py 配置驱动入口：分类/分割 demo 端到端跑通，检测构建验证通过。
  - 新增 tests/test_registry，全量 47 passed。
- [x] 阶段 2：Model 层组件化重构
  - 强化 BaseModel 契约（TASK_TYPES 校验、init_weights、from_config）。
  - TimmBackbone 支持 features_only 多尺度输出 + out_channels。
  - 新增 necks/FPN、heads/（ClsHead/SegHead/DetHead）、detectors/GeneralizedModel，均注册到注册表。
  - GeneralizedModel 验证了分类/分割/检测三任务的 backbone+neck+head 组合。
  - SimpleDetectionModel 标注为废弃死代码。
  - 新增 tests/test_models/test_components.py（6 个测试），全量 53 passed。
- [x] 阶段 3：Trainer 能力补强
  - Trainer 增加 amp / accumulate_grad_batches / seed / resume_from 参数。
  - _run_epoch 采用 AMP autocast + 梯度累积 + 按样本数加权聚合，on_batch_end 每步触发。
  - _has_lr_scheduler_callback 消除调度器双轨；fit 起始自动 resume。
  - save/load_checkpoint 纳入 scaler_state_dict 与 callbacks_state_dict；EarlyStopping / ModelCheckpoint 实现 state_dict。
  - Trainer 内 print 全部改走 utils.logger（dvisionix.trainer 命名空间）。
  - tools/train.py 增加 --resume 命令行开关并透传 amp/accumulate/seed。
  - 新增 tests/test_training/test_stage3.py（AMP smoke / 累积 / seed 可复现 / resume / 加权聚合），全量 58 passed。
- [x] 阶段 4：数据管线澄清（P2 部分完成）
  - BaseTransform 引入 provides_normalization 标记；Normalize/DetectionNormalize/SegmentationNormalize 声明为 True，Compose 及各 pipeline 自动汇总。
  - ClassificationTransforms/DetectionTransforms/SegmentationTransforms 默认自带 ImageNet 归一化，transforms 成为归一化的唯一权威。
  - BaseDataset._standardize_format 检测 transforms 状态；未提供归一化时保留旧行为并发 DeprecationWarning。
  - DetectionResize 在缩放后 clip 到目标尺寸内并过滤退化框，与 labels 同步。
  - AlbumentationsWrapper 自动补 bbox_params（pascal_voc），并双向转换 boxes/labels。
  - 新增 tests/test_data/test_stage4.py（5 项），全量 63 passed（2 条 DeprecationWarning 为预期）。
  - 磁盘全仓扫描 79 个文件，未发现 U+FFFD/mojibake，仓库内所谓"乱码"确为 PowerShell 显示假象。
- [x] 阶段 5：Metrics / 后处理正确性
  - MetricCollection 检测分支实现真正的派发（pred_boxes/pred_scores/pred_labels + batch.boxes/labels），不再是 pass。
  - DetectionMetrics 支持 use_torchmetrics=True 后端，缺依赖时打印警告回退到内置实现；文档说明内置实现的性能取舍。
  - GIoULoss docstring 明确职责边界：正负样本匹配（assigner）由 task 侧承担，Loss 仅计算已匹配对。
  - 新增 tests/test_metrics/test_stage5.py（5 项：分类准确率 / 分割 mIoU 完美 / 检测 mAP 完美与空 / MetricCollection 检测分支），全量 68 passed。
- [x] 阶段 6：文档 / 测试 / 示例收尾
  - README 增补 v0.2.0 架构概览、config 驱动 CLI/编程用法、0.1.x→0.2.0 迁移章节。
  - demos/ 六个脚本与 verify_*.py 顶部加入迁移提示，引导到 tools/train.py 与 pytest。
  - 端到端验证：tools/train.py 分类 demo 训练 + --resume 从 checkpoint 续训均通过。
  - 最终全量测试 68 passed（2 条 DeprecationWarning 为归一化过渡路径的预期提示）。
