# 变更日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
完整的规划与缺陷清单见 [CodePlan.md](CodePlan.md)。

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
