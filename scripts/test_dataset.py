import os
from pathlib import Path

import sys
import torch  # Add this import
import numpy as np

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset_utils import create_dataset, create_dataloader
# Add these imports for the test function
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from dataset.vtab_dataset import collate_fn

def save_images(images, labels, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        try:
            if isinstance(img, torch.Tensor):
                img_np = img.numpy().transpose(1, 2, 0)
            elif isinstance(img, np.ndarray):
                img_np = img.transpose(1, 2, 0) if img.shape[0] == 3 else img
            else:
                print(f"Unexpected image type: {type(img)}")
                continue

            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            plt.imsave(output_dir / f"image_{i}_label_{label}.png", img_np)
            print(f"Successfully saved image {i}")
        except Exception as e:
            print(f"Error saving image {i}: {str(e)}")


# Add this at the end of the file
def test_vtab_dataset():
    # Assuming you have a VTAB dataset in the following path
    # Expand the ~ in the path and convert to absolute path
    data_dir = Path(os.path.expanduser("~/Downloads/vtab-1k/caltech101/train800.txt")).resolve()

    image_processor = AutoImageProcessor.from_pretrained(
        '~/Downloads/dinov2-finetune/dinov2-base')

    print(f"Looking for data file at: {data_dir}")  # Add this line for debugging

    if not data_dir.exists():
        raise FileNotFoundError(f"Data file not found: {data_dir}")

    ds_vtab = create_dataset(image_processor, 'vtab', data_dir=data_dir, split='train')

    # Test __len__
    print(f"Dataset size: {len(ds_vtab)}")

    # Test __getitem__ with integer index
    img, label = ds_vtab[0]
    print(f"Single item - Image shape: {img.shape}, Label: {label}")

    # Test __getitem__ with slice
    items = ds_vtab[0:5]
    print(f"Slice - Number of items: {len(items)}")

    # Display images
    images, labels = zip(*items)
    save_images(images, labels, "output_images")

    # Test with DataLoader
    dataloader = create_dataloader(ds_vtab, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(f"Batch - Images shape: {batch['pixel_values'][0].shape}, Labels shape: {batch['labels'].shape}")


if __name__ == "__main__":
    test_vtab_dataset()
