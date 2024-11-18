# DVisionix

DVisionix 是一个基于 PyTorch 的深度学习训练框架，旨在简化计算机视觉任务的数据集构建、模型搭建和训练过程。无论你是深度学习的初学者还是经验丰富的研究人员，DVisionix 都能帮助你快速实现你的项目。

该项目目前处于开发阶段。

## 特性

- **多任务支持**：
  - 图像分类
  - 目标检测
  - 语义分割
  - 点云处理

- **预置数据集**：
  - 分类：ImageNet、CIFAR、CUB200
  - 检测：COCO、Pascal VOC
  - 分割：ADE20K、Cityscapes、Pascal VOC
  - 点云：ModelNet、S3DIS、ScanNet、SegKITTI

- **模型架构**：
  - 主干网络：SwinTransformer 等
  - 多种任务专用头部网络
  - 可扩展的模块化设计

- **训练支持**：
  - 灵活的数据增强
  - 多种评估指标
  - 可视化工具
  - 分布式训练


## 项目结构

以下是 DVisionix 项目的结构：

```
DVisionix/
│
├── docs/                    # 文档
├── logs/                    # 日志文件
├── notebooks/               # Jupyter Notebook 示例
├── src/                     # 主框架代码
│   ├── model/               # 模型定义模块
│   │   ├── backbone/        # 模型骨干网络
│   │   ├── head/            # 模型头部
│   │   │   ├── det_head/    # 检测头
│   │   │   ├── cls_head/    # 分类头
│   │   │   ├── seg_head/    # 分割头
│   │   ├── neck/            # 模型颈部
│   ├── dataset/             # 数据集处理模块
│   │   ├── classification/  # 分类数据集
│   │   ├── detection/       # 检测数据集
│   │   ├── segmentation/    # 分割数据集
│   │   ├── pointcloud/      # 点云数据集
│   ├── builder/             # 构建模块
│   ├── trainer/             # 训练逻辑模块
│   ├── utils/               # 工具函数模块
│   ├── configs/             # 配置文件
│   ├── visualizers/         # 可视化模块
│   ├── evaluators/          # 评估模块
├── tests/                   # 单元测试
├── weights/                 # 权重文件
├── requirements.txt         # 依赖包
├── README.md                # 项目说明
└── setup.py                 # 安装脚本
```

## 安装

1. 克隆项目：
   ```
   bash
   git clone https://github.com/LAZYzhaoyang/DVisionix.git
   cd DVisionix
   ```

2. 安装依赖：
   ```
   bash
   pip install -r requirements.txt
   ```

## 使用

### 数据集构建

使用 DVisionix 提供的数据集模块，轻松构建和加载数据集。

### 模型搭建

通过定义模型类，使用 PyTorch 的 `nn.Module` 轻松构建自定义模型。

### 训练模型

使用 DVisionix 的训练模块，快速启动训练过程。可以通过配置文件调整超参数。

## 示例

在 `examples/` 目录中，你可以找到一些示例代码，展示如何使用 DVisionix 进行训练和推理。

## 支持的数据集

### 分类数据集
- ImageNet：1000类图像分类数据集
- CIFAR：CIFAR-10/100 数据集
- CUB200：细粒度鸟类分类数据集

### 检测数据集
- COCO：大规模目标检测数据集
- Pascal VOC：通用目标检测数据集

### 分割数据集
- ADE20K：场景解析数据集
- Cityscapes：城市场景分割数据集
- Pascal VOC：语义分割数据集

### 点云数据集
- ModelNet：3D形状分类数据集
- S3DIS：室内场景点云分割数据集
- ScanNet：室内重建数据集
- SegKITTI：自动驾驶场景点云数据集

## 贡献指南

欢迎提交 Issue 和 Pull Request！在提交代码前，请确保：

1. 代码风格符合项目规范
2. 添加必要的文档和注释
3. 通过所有测试用例

## 许可证

本项目采用 MIT 许可证，详细信息请查看 [LICENSE](LICENSE) 文件。

## 联系

如有任何问题或建议，请联系 [zhaoyang.lee@outlook.com](mailto:zhaoyang.lee@outlook.com)。