from ..base_dataset import BaseDataset
from typing import Dict, Any, List, Tuple
import numpy as np

class SegDataset(BaseDataset):
    """Base class for segmentation datasets"""
    
    def __init__(self, *args, **kwargs):
        # Store category information
        self.categories: List[str] = []
        # Index for ignored labels in evaluation
        self.ignore_index: int = 255
        # Store label colormap for visualization
        self.label_map: Dict[int, Tuple[int, int, int]] = {}
        super().__init__(*args, **kwargs)
    
    def encode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to training format"""
        return mask
    
    def decode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to visualization format"""
        return mask