from pathlib import Path
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Dict
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor,
)
import torch
from torch.utils.data import DataLoader

from dataset.base_dataset import BaseDataset

# ref: https://github.com/dongzelian/SSF/blob/main/data/vtab.py

# dataset=("vtab-cifar(num_classes=100)" "vtab-caltech101" "vtab-dtd" "vtab-oxford_flowers102" "vtab-oxford_iiit_pet" "vtab-svhn" "vtab-sun397" "vtab-patch_camelyon" "vtab-eurosat" "vtab-resisc45" 'vtab-diabetic_retinopathy(config="btgraham-300")' 'vtab-clevr(task="count_all")' 'vtab-clevr(task="closest_object_distance")' "vtab-dmlab" 'vtab-kitti(task="closest_vehicle_distance")' 'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)' 'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)' 'vtab-smallnorb(predicted_attribute="label_azimuth")' 'vtab-smallnorb(predicted_attribute="label_elevation")')
# number_classes=(100 102 47 102 37 10 397 2 10 45 5 8 6 6 4 16 16 18 9)

VTAB_NUM_CLASSES: Dict[str, int] = {
    'cifar': 100,
    'caltech101': 102,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'svhn': 10,
    'sun397': 397,
    'patch_camelyon': 2,
    'eurosat': 10,
    'resisc45': 45,
    'dmlab': 6,
    'dsprites_loc': 16,
    'dsprites_ori': 16,
    'smallnorb_azi': 18,
    'smallnorb_ele': 9
}


@BaseDataset.register("vtab")
class VTABDataset(BaseDataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.root_dir = self.data_dir.parent.parent
        self.task = self.data_dir.parent.name
        self.transform = transform if transform is not None else ToTensor()
        self.samples, self.labels = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_dir}")

        task_dir = self.root_dir / self.task
        
        with self.data_dir.open('r') as f:
            lines = f.read().splitlines()
        
        samples, labels = zip(*(line.split(maxsplit=1) for line in lines))
        
        return (
            np.array([task_dir / sample for sample in samples]),
            np.array(labels, dtype=int)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Image.Image, int], List[Tuple[Image.Image, int]]]:
        if isinstance(idx, slice):
            return [self.get_single_item(i) for i in range(*idx.indices(len(self)))]
        return self.get_single_item(idx)

    def get_single_item(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, label = self.samples[idx], self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        return image, int(label)  # Ensure label is an integer


def make_transform(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

    # crop_size = (image_processor.crop_size["height"], image_processor.crop_size["width"])
    train_transform = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    eval_transform = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

    return dict(train_transform=train_transform, eval_transform=eval_transform)


def create_dataset(name, *args, **kwargs):
    dataset_class = registry.get(name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset_class(*args, **kwargs)


def get_vtab_dataloader(
        data_dir: str,
        task: str,
        stage: str,
        batch_size: int,
        image_processor,
        num_workers: int = 1,
        shuffle: bool = True
) -> DataLoader:
    transform = make_transform(image_processor)
    dataset = VTABDataset(
        root_dir=data_dir,
        task=task,
        stage=stage,
        transform=transform[f'{stage}_transform']
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(examples: List[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
