import os
from pathlib import Path

import sys
from pathlib import Path

import numpy as np

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dataset.base_dataset import BaseDataset
# Add these imports for the test function
import matplotlib.pyplot as plt


# Add this at the end of the file
def test_vtab_dataset():
    # Assuming you have a VTAB dataset in the following path
    # Expand the ~ in the path and convert to absolute path
    data_dir = Path(os.path.expanduser("~/Downloads/vtab-1k/caltech101/train800.txt")).resolve()

    print(f"Looking for data file at: {data_dir}")  # Add this line for debugging

    if not data_dir.exists():
        raise FileNotFoundError(f"Data file not found: {data_dir}")

    ds_cls = BaseDataset.by_name('vtab')
    ds_vtab = ds_cls(data_dir)

    # Test __len__
    print(f"Dataset size: {len(ds_vtab)}")

    # Test __getitem__ with integer index
    img, label = ds_vtab[0]
    print(f"Single item - Image shape: {img.shape}, Label: {label}")

    # Test __getitem__ with slice
    items = ds_vtab[0:5]
    print(f"Slice - Number of items: {len(items)}")

    # Visualize a few images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, (img, label) in enumerate(items):
        # Convert tensor to numpy array and ensure it's in the correct range
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)  # Ensure values are between 0 and 1

        axes[i].imshow(img_np)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Test with DataLoader
    # dataloader = DataLoader(ds_vtab, batch_size=32, shuffle=True)
    # batch = next(iter(dataloader))
    # print(f"Batch - Images shape: {batch[0].shape}, Labels shape: {batch[1].shape}")


if __name__ == "__main__":
    test_vtab_dataset()
