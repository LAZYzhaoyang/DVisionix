from ..base_dataset import BaseDataset
from typing import Dict, Any, List

class ClsDataset(BaseDataset):
    """Base class for classification datasets"""
    
    def __init__(self, *args, **kwargs):
        # Store category information for classification
        self.categories: List[str] = []
        # Store class-to-index mapping
        self.class_to_idx: Dict[str, int] = {}
        super().__init__(*args, **kwargs)
    
    def get_categories(self) -> List[str]:
        """Return the list of categories"""
        return self.categories