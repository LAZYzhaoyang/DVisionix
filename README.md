<div align="center">

# 🔬 DVisionix

**一个模块化、可扩展的深度学习算法库

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ✨ 核心特性

### 🏆 真正通用的训练架构
- **Task 系统**：任务逻辑与训练循环完全解耦
- **Callback 系统**：灵活的生命周期钩子
- **通用 Trainer**：支持任意任务，不需要修改训练代码

### 📦 统一的数据接口
- 所有任务（分类、检测、分割）使用相同的 BaseDataset 基类
- 统一的字典格式输出
- build_dataset 配置驱动构建数据集（注册即用）

### 📊 完整的指标支持
- 分类：Accuracy, Precision, Recall, F1
- 分割：mIoU, Pixel Accuracy, Mask mAP（实例/掩码评估）
- 检测：COCO-style mAP@0.5, mAP@0.5:0.95

### 🎯 内置任务支持
- ✅ 图像分类
- ✅ 目标检测（COCO/VOC 格式）
- ✅ 语义分割（Cityscapes/ADE20K）

---

### 🏗️ 架构概览（v0.7.1）

```
dvisionix/
├── registry.py          # 全局注册表（MODELS / TASKS / LOSSES / METRICS / DATASETS / ...）
├── config/               # Config（YAML 继承 / CLI 覆盖 / schema 校验）
├── models/
│   ├── base.py          # BaseModel（TASK_TYPES 校验 / init_weights / from_config）
│   ├── layers/          # 自定义层 + timm 层封装（ConvNormAct / SE / MLP / DropPath / CSP / ELAN / E-ELAN / 可变形注意力）
│   ├── backbones/       # Timm / Sequential / ConvNeXt(+V2) / CSPDarknet / MobileNetV3 / EfficientNetLite / ViT / Swin / MiT(SegFormer)
│   ├── necks/           # FPN / PANet
│   ├── heads/           # 分类（Cls/ArcFace/CosFace/SphereFace/AdaFace/MultiLabel/NormFace/CurricularFace/PartialFC/CircleLoss/SimCLR）· 分割（Seg/FCN/DeepLabV3(+)/UNet/SegFormer(+V2)/MaskFormer(+2)/PSP/UPerNet/BiSeNet/SwinUNet）· 检测（Det/FCOS/RetinaNet/YOLO(v8/v10)/DETR/RT-DETR(+full)/DeformableDETR/CenterNet）
│   ├── detectors/       # SingleStageDetector + FCOS / RetinaNet / YOLO(v8/v9/v10) / DETR / RT-DETR(+full) / DeformableDETR / CenterNet（decode 与模型同文件）
│   ├── classifiers/     # 分类组合器子包（LinearClassifier）
│   ├── segmenters/      # 分割组合器子包（SegmentationModel / SwinUNet）
│   ├── toy/             # 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）
│   └── losses/          # BaseLoss + LossComposer + 检测 assigner/损失（即插即用）
├── training/
│   ├── trainer.py       # 统一 Trainer（Task 驱动 / DDP / AMP / 梯度累积 / resume / work_dir）
│   ├── tasks/           # BaseTask + 分类/检测/分割任务（optimizer/loss/metrics 全配置化）—— 主扩展点
│   ├── callbacks/       # CallbackList + ProgressBar/ModelCheckpoint/EarlyStopping —— 主扩展点
│   ├── optim/           # OPTIMIZERS / SCHEDULERS 注册表 + build_optimizer / build_scheduler
│   ├── workdir.py       # 工作目录隔离 + resume 三态 + config dump
│   └── builder.py       # build_callbacks / build_trainer
├── data/
│   ├── sample.py         # Sample 协议 + ImageMode / ImageInfo / NormalizationSpec
│   ├── base.py          # BaseDataset（Sample 驱动 + mask 路径加载）
│   ├── collate.py       # detection_collate / segmentation_collate
│   ├── presets.py       # 主流公开数据集工具箱（CIFAR/ImageNet/COCO/VOC/Cityscapes/ADE20K/ImageFolder）
│   ├── datasets/
│   │   └── custom.py        # CustomDataset（自定义模板范例）
│   └── transforms/
│       ├── base.py        # BaseTransform / TransformPipeline
│       ├── image.py       # 原子：ImageResize / Flip / Crop / ColorJitter / Normalize / ToTensor
│       ├── geometric.py   # 几何同步：BoxSyncResize / BoxSyncRandomHorizontalFlip / BoxSyncRandomCrop / BoxSyncPad
│       ├── labels.py      # LabelToTensor / BoxesToTensor / MaskToTensor
│       ├── third_party.py # AlbumentationsWrapper（适配 albumentations / kornia / torchvision）
│       └── builder.py     # build_transform / build_pipeline
├── metrics/
│   ├── classification.py  # Accuracy / Precision / Recall / F1
│   ├── segmentation.py    # mIoU / Pixel Accuracy
│   ├── detection.py       # COCO mAP（内置实现 + torchmetrics 可选）
│   ├── collection.py      # MetricCollection 组合容器（原子指标自由组合）
│   └── presets.py         # get_preset_metrics + 预设组合类
└── utils/
    └── logging/           # 日志/可视化（console + file + JSONL + TensorBoard）
```

- **配置驱动**：所有入口统一通过 `tools/train.py --config` 或 `Config` 编程 API。
- **注册表**：`build_model` / `build_task` / `build_loss` / `build_metric` / `build_dataset` 从全局注册表按名称查找。
- **归一化唯一权威**：transforms 内的 `Normalize` 类掌控归一化，`BaseDataset` 自动感知避免二次归一化。
- **Loss 在模型层**：dvisionix/models/losses（BaseLoss 继承 + LossComposer 自由组合，接入任意 Task）。
- **日志/可视化**：统一在 utils/logging（TrainingLogger），支持 console / 文件 / JSONL / TensorBoard。
- **工作目录与续训**：默认 ~/dvisionix_runs/&lt;exp&gt;/&lt;ts&gt;（代码库外），--resume auto 自动续训；支持 DDP 多卡。

---

## 🚀 快速开始


### 安装

`ash
# 克隆项目
git clone https://github.com/LAZYzhaoyang/DVisionix.git
cd DVisionix

# 创建 conda 环境
conda create -n dvisionix python=3.10 -y
conda activate dvisionix

# 安装 PyTorch（根据你的 CUDA 版本）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
pip install opencv-python numpy pyyaml tensorboard matplotlib
`

### 3 分钟上手：Config 驱动训练

```bash
# 使用合成数据快速验证分类 pipeline
python tools/train.py --config configs/classification/demo_synthetic.yaml

# 覆盖单个参数
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --cfg-options training.num_epochs=5 training.optimizer.lr=0.01

# 自动续训最近一次任务（work_dir 内自动查找 last.pt）
python tools/train.py --config configs/classification/demo_synthetic.yaml \
  --resume auto
```

**编程方式：**

```python
from dvisionix.config import Config
from dvisionix.models import build_model
from dvisionix.training import Trainer, build_task

cfg = Config.from_yaml("configs/classification/demo_synthetic.yaml")
model = build_model(cfg.model.to_dict())
task = build_task(cfg.task.to_dict() if "task" in cfg else {"type": "ClassificationTask", "num_classes": cfg.model.num_classes})
trainer = Trainer(task=task, train_loader=train_loader, val_loader=val_loader,
                  max_epochs=cfg.training.num_epochs, amp=True, seed=42)
trainer.fit(model)
```


---

## 📖 文档索引

| 文档 | 说明 |
|------|------|
| [快速开始](docs/quick_start.md) | 安装、Config 驱动训练、编程式训练 |
| [配置系统](docs/config_system.md) | Config 加载 / 继承 / CLI 覆盖 / schema 校验 |
| [数据模块](docs/data.md) | Sample 协议、BaseDataset、transforms、数据集工具箱 |
| [骨干网络（timm）](docs/backbones.md) | TimmBackbone / TimmClassifier |
| [自定义 Layer 与 Model](docs/custom_models.md) | layers、注册与配置驱动组装 |
| [指标（Metrics）](docs/metrics.md) | 原子指标、MetricCollection、预设组合 |
| [日志系统](docs/logging.md) | TrainingLogger / JSONL / TensorBoard |
| [模型导出（ONNX）](docs/model_export.md) | ONNXExporter 导出与精度验证 |
| [目标检测](docs/detection.md) | 网格检测器、collate、mAP 评估 |
| [语义分割](docs/segmentation.md) | 分割数据格式与端到端训练 |

---

## 🏗️ 架构设计

### 核心组件

`
┌─────────────────────────────────────────────────────────────┐
│                    Trainer (通用训练引擎)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │  Epoch Loop  │  │  Callback  │  │  Metrics  │ │
│  │  Batch Loop  │  │  System   │  │  Collector │ │
│  └───────────┘  └───────────┘  └───────────┘ │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│                   Task (任务逻辑)                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────┐ │
│  │ training_step  │  │ validation_step  │  │ configure │ │
│  │   (forward + loss) │  │  (eval only)    │  │  optimizer  │ │
│  └───────────────┘  └───────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
`

### 模块依赖

`
data (BaseDataset, build_dataset, Transforms)
    ↓
models (BaseModel, CNN, Segmentation, Detection, Losses)
    ↓
training (Trainer, Tasks, Callbacks)
    ↓
metrics (原子指标 + MetricCollection + 预设组合)
`

---

## 🎯 自定义任务示例

DVisionix 最强大的功能是支持完全自定义的任务逻辑。只要实现三个方法，就可以用通用 Trainer 训练：

`python
from dvisionix.training import BaseTask, Trainer

class MyAIModelTask(BaseTask):
    """自定义 AI 任务"""
    
    def __init__(self, my_param=0.5):
        super().__init__()
        self.my_param = my_param
        self.loss_fn = YourCustomLoss()
    
    def training_step(self, model, batch, device):
        """自定义单步训练逻辑"""
        # 获取数据
        x = batch["your_data"].to(device)
        target = batch["your_target"].to(device)
        
        # 前向传播
        output = model(x)
        
        # 计算损失（你想怎么算就怎么算
        loss = self.loss_fn(output, target)
        
        # 你可以返回任意指标，都会被自动记录
        return {
            "loss": loss,
            "your_metric1": metric1,
            "your_metric2": metric2,
        }
    
    def validation_step(self, model, batch, device):
        """自定义单步验证逻辑"""
        x = batch["your_data"].to(device)
        target = batch["your_target"].to(device)
        
        with torch.no_grad():
            output = model(x)
            loss = self.loss_fn(output, target)
        
        return {"loss": loss}
    
    def configure_optimizers(self, model):
        """自定义优化器配置"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        
        # 支持多种返回格式
        return optimizer  # 或 (optimizer, scheduler)
`

然后直接用 Trainer 训练：

`python
trainer = Trainer(
    task=MyAIModelTask(my_param=0.7),
    train_loader=your_train_loader,
    val_loader=your_val_loader,
    max_epochs=100,
)

trainer.fit(your_model)
`

---

## 📁 项目结构

```
DVisionix/
├── dvisionix/                  # 核心库代码
│   ├── registry.py             # 全局注册表 + build_from_cfg
│   ├── config/                 # Config（YAML 继承 / CLI 覆盖）
│   ├── data/                   # BaseDataset / Sample 契约 / transforms / presets / collate
│   ├── models/                 # BaseModel / backbones / necks / heads / detectors
│   │   └── losses/             # Loss 组件（BaseLoss 继承 + LossComposer 组合）
│   ├── training/               # Trainer / tasks / callbacks / optim / workdir
│   ├── metrics/                # 原子指标 + MetricCollection 组合
│   ├── utils/logging/          # 日志/可视化（console + file + JSONL + TensorBoard）
│   └── export/                 # ONNX 导出
├── tools/train.py              # 配置驱动训练入口
├── configs/                    # 任务示例配置
├── tests/                      # pytest 测试
└── demos/                      # 演示脚本
```

---

## 🧪 测试

`ash
# 运行所有测试
pytest tests/ -v

# 只运行训练模块测试
pytest tests/test_training/ -v

# 只运行数据模块测试
pytest tests/test_data/ -v
`

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 📌 引用

如果你觉得这个项目对你有帮助，欢迎给个 Star ⭐

---

## 📞 问题反馈

如有问题或建议，请在 GitHub 提交 Issue。

---

## 📌 版本演进

### 0.17.0（训练工程 P2：超参搜索 / 特征蒸馏）
- 超参搜索工具 `tools/hparam_search.py`：YAML 参数网格（点路径 -> 候选值）笛卡尔积或随机采样，
  逐 trial 独立子进程跑 `train.py` 并汇总 `search_results.csv`（示例 `configs/classification/hparam_search.yaml`）；
- 特征蒸馏：`FeatureDistillLoss`（`feature_distill`，MSE + 可选 L2 归一化，支持多层特征）+ `DistillCallback` 新增 `feature_extractor`；
- 新增 P2 测试 3 项，全量 271 passed + 2 skipped；ruff / black 全绿。

### 0.16.0（DINO-lite / 线性评估 / 训练工程 P1）
- 检测：`dinodetr`（hybrid query selection + query denoising 对比正负样本 + box refinement）；
- 自监督闭环：`LinearEvalTask`（冻结 encoder + L2 归一化线性头）+ `load_backbone` + `pretrained_backbone` 配置；
- 训练工程：`linear_warmup` 调度器、`gradient_clip_value`、EMA `decay_warmup_epochs` + 最终导出 `ema_last.pt`；
- 新增 dino / linear_eval 配置示例；全量测试 271 passed + 2 skipped；ruff / black 全绿。

### 0.15.0（组合器目录化 + 批次 2）
- classifiers / segmenters 子包化（顶层 API 不变，调用规则新增 R7）；
- `swinv2_backbone`（cosine attention + 连续相对位置偏置 + res-post-norm）；
- `multi_scale_deformable_attention_v2`（分层参考点 + 尺度归一采样偏移）；
- `segformer_v3_head`（SE 融合解码）；
- 全量测试 265 passed + 2 skipped；ruff / black 全绿。

### 0.14.0（中期批次 1：骨干/分割/检测扩展）
- 骨干：`convnextv2_backbone`（GRN）、`efficientnet_lite_backbone`（MBConv+SE）、`mit_backbone`（SegFormer encoder）；
- 分割：`swin_unet` 组合模型（Swin encoder + PatchExpand 解码 + 跳连）；
- 检测：`c3k2_block` / `psa_block`（YOLOv11 风格骨干组件）+ yolov11 配置示例；
- 新增 3 个配置示例；全量测试 260 passed + 2 skipped；ruff / black 全绿。

### 0.13.0（model 模块分层与 layers 统一重构）
- 共用子模块下沉：`LayerNorm2d` / `DeformableEncoder(Decoder)Layer` / `MixFFN` / `WindowAttention` / `PatchMerging` / `PatchExpand` 统一到 layers；
- `anchors` 下沉到 layers（消除 losses→detectors 反向依赖）；`PixelDecoder` 入 necks 与 FPN 同级；共享解码器 `maskformer_decode` 入 postprocess；
- 确立并写入 CodePlan「model 模块调用规则 R1-R6」（分层调用图 + 各层职责表，单向依赖 + 同级隔离）；
- 全量测试 250 passed + 2 skipped；ruff / black 全绿。

### 0.12.0（Transformer 骨干 / YOLOv9-lite）
- 骨干：`vit_backbone`（patch embed + Transformer encoder，正弦位置编码支持任意尺寸）、
  `swin_backbone`（window/shifted-window 注意力 + PatchMerging，stride 4/8/16/32）；
- YOLOv9-lite：`ReversibleBlock`（可逆 + 逆变换）+ `yolo_v9`（PGI 辅助头，仅训练）+ `yolo_v9_detection` 损失 + 配置示例；
- 多卡实验与 torchmetrics 迁移详细计划已**永久写入 CodePlan**（默认推迟，仅按明确指示实施）；
- 全量测试 250 passed + 2 skipped；ruff / black 全绿。

### 0.11.0（骨干库 / SimCLR / 分割增强）
- 内置骨干：`convnext_backbone` / `cspdarknet_backbone` / `mobilenetv3_backbone`（通用 `FeatureBackboneBase`，
  新增 `ConvNeXtBlock` / `MBConvBlock` 层），分类/检测/分割即插即用；
- SimCLR 端到端：`SimCLRTransforms`（双视角）+ `SimCLRTask`（InfoNCE）+ `tools/train.py` 支持 simclr 任务 + 配置示例；
- 分割：`segformer_v2_head`（overlap patch embed + MixFFN）、`swin_unet_decoder`（PatchExpand 上采样 + 跳连）；
- YOLOv9-lite（PGI）详细计划已写入 CodePlan，待实施；ViT/Swin 骨干列为后续；
- 全量测试 244 passed + 2 skipped；ruff / black 全绿。

### 0.10.0（方向 3：模型继续扩充）
- YOLO 系列：`EELANLayer`（YOLOv7 风格骨干）、`yolo_v10`（NMS-free：one-to-one 损失 + 免 NMS 解码）；
- CenterNet：`centernet`（关键点热图 + 宽高 + 偏移，penalty-reduced Focal + L1）；
- 分割：`bisenet_head`（轻量实时分割）；
- 分类：`circle_loss_head` + `CircleLoss`、`simclr_head` + `InfoNCELoss`（对比学习）；
- 新增 yolov7 / yolov10 / centernet 配置示例；全量测试 236 passed + 2 skipped；ruff / black 全绿。

### 0.9.0（阶段 C：推荐组合实施）
- 全景评估接入验证循环：`MaskFormerTask(panoptic=True)` 输出 `PQ / SQ / RQ`（`PanopticQuality`）；
- RT-DETR 增强版：`rtdetr_full`（多尺度可变形编码器 + IoU-aware query selection + 框细化，保留 compact 版）；
- Mask2Former 完整版：`mask2former_head`（mask attention 解码器 + FPN 像素解码器，逐层细化掩码）；
  `MaskFormerLoss` 支持真实实例 GT（`instance_masks` / `instance_labels`）；
- 修复 mask mAP 真值格式（List[(M,H,W)]）与分辨率对齐；
- 全量测试 229 passed + 2 skipped；ruff / black 全绿。

### 0.8.0（阶段 B：模型库扩充）
- 分割：新增 `PSPHead` / `UPerNetHead` / `DeepLabV3PlusHead`（即插即用，多尺度头走 input_style）；
  `MaskFormerTask` 实例分割训练任务；`PanopticQuality`（PQ/SQ/RQ）+ `panoptic_decode` / `evaluate_panoptic` 打通全景评估；
- 检测：新增 `CSPLayer` / `ELANLayer`（YOLOv5/v9 风格骨干即插即用，附配置示例）；
  `DeformableDETRHead` / `DeformableDETRDetector`（纯 PyTorch 多尺度可变形注意力，复用 DETRLoss/decode）；
- 分类：新增 `NormFaceHead` / `CurricularFaceHead` / `PartialFCHead`（大规模类别采样 softmax）；
- 全量测试 223 passed + 2 skipped；ruff / black 全绿。

### 0.7.1（decode 归位 / 组合性验证）
- 模型专属 decode 从共享 `postprocess.py` 移回各模型文件：`detr_decode`（detectors/base.py，DETR/RT-DETR 共用）、
  `fcos_decode`（detectors/fcos.py）、`retinanet_decode`（detectors/retinanet.py）、`yolo_decode`（detectors/yolo.py）、
  `maskformer_decode`（heads/segmentation/maskformer.py）；`postprocess.py` 只保留共享原语 `nms / batched_nms / box_iou`；
- 顶层 API 保持兼容（`dvisionix.models.*_decode` 仍可直接导入）；
- heads 与 backbone/neck 组合性冒烟验证 27 组全部通过（FCOS/RetinaNet/YOLO/DETR/RT-DETR × Sequential/Timm × 无 neck/FPN/PANet，
  分割 6 头、分类 6 头同样验证）；
- 文档同步：CodePlan 增补 v0.7.1 章节；docs/detection.md 重写为组件化检测器；segmentation/custom_models/docs 索引更新。
- 阶段 A 收口：`MaskFormerHead` / `SegmentationModel` 新增 `.decode()`（统一解码契约）；多尺度头
  `input_style="multi_scale"` 自声明（装配器统一注入，删除硬编码名单）；RT-DETR 支持 FPN/PANet neck；
  分类头补 `_head` 注册别名（旧名兼容）；新增 6 项测试，全量 210 passed + 2 skipped。

### 0.7.0（RT-DETR / mask mAP / 本地 lint 工具链）
- 检测：新增 `RTDETRDetector`（RT-DETR-lite：混合编码器 + query 选择 + 解码器，复用 DETRLoss/decode）；
- 实例分割：新增 `MaskAveragePrecision`（mask mAP，COCO 风格）+ `maskformer_decode` + `evaluate_mask_ap`；
- 本地工具链落地：安装并跑通 ruff（0 错误）+ black（全仓格式化，CI 一致）；
- 回调 `on_batch_begin/end` 支持透传 batch（供 DistillCallback 使用）。

### 0.6.0（训练工程与 Mask2Former 完整版）
- MaskFormer 完整版：mask 监督（匈牙利 mask 匹配 + CE/Dice/BCE，`MaskFormerLoss`）；
- 训练工程：`EMA` 回调、`DistillCallback` + `DistillationLoss`、ModelCheckpoint 保留策略
  （`save_every_n_epochs` / `max_epoch_checkpoints`）、`history.csv` 导出；
- CI：GitHub Actions（ruff / black / pytest，CPU 矩阵 3.10/3.11）；
- 回调新增 `on_validation_begin/end` 钩子；YOLO 配置示例。

### 0.5.0（前沿检测与分割）
- 检测：新增 `YOLODetector`（YOLOv8 风格，TaskAlignedAssigner）与 `DETRDetector`（transformer + HungarianMatcher）；
- 分割：新增 `SegFormerHead`（MLP 解码）与 `MaskFormerHead`（query 掩码解码，compact）；
- 分类：新增 `CosFaceHead` / `SphereFaceHead` / `AdaFaceHead`（度量学习）与 `MultiLabelTask`；
- heads 模块按任务重组为子包（classification / segmentation / detection），每头一个文件；
- 修复 Hungarian 匹配器 n>m 时死循环（方阵补齐）。

### 0.4.0（模型模块丰富）
- 删除 GeneralizedModel（三任务万能模型，契约弱、检测半成品），替换为具体模型 + 共享脚手架；
- 检测：`FCOSDetector`（anchor-free）与 `RetinaNetDetector`（anchor-based）并存，
  assigner 即插即用（FCOS / MaxIoU / ATSS）；
- 分割：`SegmentationModel` + `DeepLabV3Head`(ASPP) / `UNetDecoder` / `FCNHead`；
- 分类：`LinearClassifier` + `ArcFaceHead` / `MultiLabelHead`（+ BCE 多标签损失）；
- 颈部：新增 `PANet`（FPN + 自底向上路径）；
- 教学模型（SimpleCNN / SimpleSegmentationModel / GridDetectionModel）迁入 `models.toy`；
- 配置系统支持 `_delete_: true` 替换语义（解决继承时类型专属参数残留）。
- `training/` 重组为 `tasks/` `callbacks/` `optim/` 子包（顶层 API 不变）；
- Config 新增 schema 校验（未知键告警 / 类型校验），CLI 支持 list/dict；
- `ONNXExporter` 支持多输入 / 多输出 / dict 输出 / `backend=trace|dynamo` / 归一化元数据。

### 0.3.0（训练子系统重构）
破坏性变更：
- **Loss 迁移到模型层**：`dvisionix.models.losses`（`BaseLoss` 继承 + `LossComposer` 组合），删除 `dvisionix.training.losses`。
- **Task 全面配置化**：optimizer / scheduler / loss / metrics 均由配置驱动；
  `TensorBoardLogger`、`LearningRateScheduler` 回调与 `utils.visualization.Visualizer` 已移除，日志统一走 `utils.logging.TrainingLogger`。
- **训练能力**：验证指标（acc / mAP / mIoU）进入 epoch 日志；DDP 多卡；work_dir 隔离（默认 `~/dvisionix_runs/<exp>/<ts>`，代码库外）；`--resume auto` 自动续训。

### 0.2.0（组件化重构）
- 引入全局注册表与配置驱动入口 `tools/train.py`；模型组件化（backbone/neck/head）；
  `BaseDataset` 归一化交给 transforms（唯一权威）。
