# D:\\ZhaoyangProject\\DVisionix\\tests\\test_training\\test_trainer.py

"""
测试通用训练引擎和任务系统
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dvisionix.training import (
    Trainer,
    ClassificationTask,
    SegmentationTask,
    BaseTask,
    ProgressBar,
    ModelCheckpoint,
)
from dvisionix.models import SimpleCNN, SimpleSegmentationModel


class TestTaskSystem:
    """测试任务系统"""
    
    def test_classification_task(self):
        """测试分类任务"""
        task = ClassificationTask(num_classes=10, learning_rate=1e-3)
        
        # 测试 configure_optimizers
        model = SimpleCNN(num_classes=10)
        opt_config = task.configure_optimizers(model)
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        
        # 测试 training_step
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.randint(0, 10, (4,)),
        }
        device = torch.device("cpu")
        
        result = task.training_step(model, batch, device)
        assert "loss" in result
        assert "acc" in result
        assert isinstance(result["loss"], torch.Tensor)
    
    def test_custom_task(self):
        """测试自定义任务"""
        
        class MyCustomTask(BaseTask):
            def __init__(self):
                super().__init__()
                self.loss_fn = torch.nn.MSELoss()
            
            def training_step(self, model, batch, device):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = self.loss_fn(pred, y)
                return {"loss": loss}
            
            def validation_step(self, model, batch, device):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                loss = self.loss_fn(pred, y)
                return {"loss": loss}
            
            def configure_optimizers(self, model):
                return torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 简单的回归模型
        class RegressionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        task = MyCustomTask()
        model = RegressionModel()
        
        # 测试优化器配置
        optimizer = task.configure_optimizers(model)
        assert isinstance(optimizer, torch.optim.Optimizer)
        
        # 测试训练步
        batch = {"x": torch.randn(4, 10), "y": torch.randn(4, 1)}
        device = torch.device("cpu")
        
        result = task.training_step(model, batch, device)
        assert "loss" in result


class TestCallbacks:
    """测试回调系统"""
    
    def test_progress_bar(self):
        """测试进度条回调"""
        callback = ProgressBar(log_interval=10)
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_batch_end")
    
    def test_model_checkpoint(self, tmp_path):
        """测试模型检查点回调"""
        callback = ModelCheckpoint(save_dir=str(tmp_path), monitor="val_loss", mode="min")
        assert hasattr(callback, "on_epoch_end")


class TestTrainer:
    """测试通用训练引擎"""
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        # 创建简单的数据集
        x = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        
        # 包装为字典格式的数据集
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return {"image": self.x[idx], "label": self.y[idx]}
        
        dataset = DictDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # 创建任务和训练器
        task = ClassificationTask(num_classes=10)
        trainer = Trainer(task, train_loader, max_epochs=2)
        
        assert trainer.max_epochs == 2
        assert trainer.task is task
    
    def test_trainer_fit(self):
        """测试训练流程"""
        # 小数据集快速测试
        x = torch.randn(20, 3, 32, 32)
        y = torch.randint(0, 10, (20,))
        
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return {"image": self.x[idx], "label": self.y[idx]}
        
        train_dataset = DictDataset(x, y)
        val_dataset = DictDataset(x[:10], y[:10])
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        model = SimpleCNN(num_classes=10)
        task = ClassificationTask(num_classes=10)
        
        trainer = Trainer(
            task,
            train_loader,
            val_loader,
            max_epochs=1,
            log_interval=10,
        )
        
        result = trainer.fit(model)
        
        assert result is not None
        assert trainer.current_epoch == 0  # 0-based


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
