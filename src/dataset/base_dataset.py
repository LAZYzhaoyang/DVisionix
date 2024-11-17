from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple

class BaseDataset(Dataset, ABC):
    """
    Base class for DVisionix datasets
    All specific task datasets should inherit from this base class and implement the necessary abstract methods.
    """
    def __init__(self, 
                 data_root: str,
                 transform: Optional[Any] = None,
                 mode: str = 'train'):
        """        
        Parameters:
            data_root (str): dataset root directory
            transform: data augmentation/preprocessing pipeline
            mode (str): dataset mode ('train', 'val', 'test')
        """        
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.data_infos = []  # 存储数据集信息的列表
        
        # 初始化数据集
        self._init_dataset()
    
    @abstractmethod
    def _init_dataset(self) -> None:
        """
        Initialize the dataset and read dataset information
        Subclasses must implement this method to load the basic information of the dataset
        """
        pass
    
    @abstractmethod
    def _load_data(self, index: int) -> Dict[str, Any]:
        """
        Load a single data sample
        Parameters:
            index (int): data index
        Returns:
            Dict[str, Any]: a dictionary containing data sample information
        """
        pass
    
    def __len__(self) -> int:
        """Returns the size of the dataset"""
        return len(self.data_infos)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single data sample
        Parameters:
            index (int): Data index
        Returns:
            Dict[str, Any]: Processed data sample
        """
        # 加载原始数据
        data = self._load_data(index)
        
        # 应用数据转换
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
    def evaluate(self, 
                results: list, 
                metrics: Optional[list] = None, 
                **kwargs) -> Dict[str, float]:
        """
        Evaluation interface
        Parameters:
            results (list): Model prediction results
            metrics (list): List of evaluation metrics
        Returns:
            Dict[str, float]: Evaluation results
        """
        raise NotImplementedError