from typing import Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor
)


def create_transform(image_processor, is_train: bool):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

    if is_train:
        transform = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
            )
    else:
        transform = Compose(
            [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

    return transform

def create_dataset(
    image_processor: AutoImageProcessor,
    dataset_name: str,
    data_dir: str,
    split: str,
    **kwargs
) -> Dataset:
    """
    Create a dataset based on the provided configuration.
    """
    
    # Import here to avoid circular imports
    from dataset.base_dataset import BaseDataset
    dataset_cls = BaseDataset.by_name(dataset_name)
    
    transform = create_transform(image_processor, is_train=split=="train")
    
    
    return dataset_cls(
        data_dir=data_dir,
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