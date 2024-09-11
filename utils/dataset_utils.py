from typing import Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import os

from dataset.base_dataset import BaseDataset

def create_dataset(
    dataset_type: str,
    data_dir: str,
    split: str,
    transform_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dataset:
    """
    Create a dataset based on the provided configuration.
    """
    data_dir = Path(os.path.expanduser(data_dir)).resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    dataset_cls = BaseDataset.by_name(dataset_type)
    
    if transform_config:
        transform = create_transform(transform_config)
    else:
        transform = None
    
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        transform=transform,
        **kwargs
    )

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )