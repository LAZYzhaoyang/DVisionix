# D:\ZhaoyangProject\DVisionix\tests\test_data\test_basic.py

"""
数据模块的基础单元测试

测试 BaseDataset、TaskType、DataFormat、CustomDataset 等基础功能。
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dvisionix.data import (
    BaseDataset,
    TaskType,
    DataFormat,
    CustomDataset,
    ClassificationTransforms,
)


def test_task_type():
    """测试 TaskType 枚举"""
    assert str(TaskType.CLASSIFICATION) == "classification"
    assert str(TaskType.DETECTION) == "detection"
    assert str(TaskType.SEGMENTATION) == "segmentation"
    
    # 测试从字符串创建
    assert TaskType("classification") == TaskType.CLASSIFICATION


def test_data_format():
    """测试 DataFormat 数据类"""
    df = DataFormat(num_classes=10)
    assert df.num_classes == 10
    assert df.image_channels == 3
    
    # 测试 to_dict
    d = df.to_dict()
    assert "num_classes" in d
    assert d["num_classes"] == 10


def create_test_image(path, size=(64, 64)):
    """创建一个测试图像"""
    image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(path, image)
    return path


def test_custom_dataset_classification():
    """测试分类任务的自定义数据集"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试图像
        img_paths = []
        for i in range(5):
            path = os.path.join(tmpdir, f"img_{i}.png")
            create_test_image(path)
            img_paths.append(path)
        
        # 创建样本列表
        samples = [
            {"image_path": p, "label": i % 2}
            for i, p in enumerate(img_paths)
        ]
        
        # 创建数据集
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
        )
        
        # 测试基本属性
        assert len(dataset) == 5
        assert dataset.get_task_type() == TaskType.CLASSIFICATION
        
        # 测试获取样本
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape[0] == 3  # CHW 格式
        assert sample["label"].item() in [0, 1]


def test_custom_dataset_with_transforms():
    """测试带变换的自定义数据集"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_paths = []
        for i in range(3):
            path = os.path.join(tmpdir, f"img_{i}.png")
            create_test_image(path, size=(100, 100))
            img_paths.append(path)
        
        samples = [
            {"image_path": p, "label": 0}
            for p in img_paths
        ]
        
        # 创建变换
        transforms = ClassificationTransforms(train=True, image_size=32)
        
        # 创建数据集
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
            transforms=transforms,
        )
        
        # 测试变换是否生效
        sample = dataset[0]
        assert sample["image"].shape[1] == 32  # H
        assert sample["image"].shape[2] == 32  # W


def test_dataset_summary():
    """测试数据集摘要信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.png")
        create_test_image(path)
        
        samples = [{"image_path": path, "label": 0}]
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
        )
        
        summary = dataset.summary()
        assert summary["num_samples"] == 1
        assert summary["task_type"] == "classification"


def test_return_meta():
    """测试元信息返回"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.png")
        create_test_image(path)
        
        samples = [{"image_path": path, "label": 0}]
        dataset = CustomDataset(
            task_type="classification",
            samples=samples,
            num_classes=2,
            return_meta=True,
        )
        
        sample = dataset[0]
        assert "meta" in sample
        assert "image_path" in sample["meta"]


if __name__ == "__main__":
    print("Running tests...")
    
    test_task_type()
    print("✓ test_task_type passed")
    
    test_data_format()
    print("✓ test_data_format passed")
    
    test_custom_dataset_classification()
    print("✓ test_custom_dataset_classification passed")
    
    test_custom_dataset_with_transforms()
    print("✓ test_custom_dataset_with_transforms passed")
    
    test_dataset_summary()
    print("✓ test_dataset_summary passed")
    
    test_return_meta()
    print("✓ test_return_meta passed")
    
    print("\n✅ All tests passed!")