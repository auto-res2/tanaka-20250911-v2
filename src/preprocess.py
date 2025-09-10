"""src/preprocess.py
Very small synthetic dataset creator used exclusively for CI / unit-testing. It
emits random images so that the training loop has data to iterate over without
needing to download ImageNet.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class RandomImageDataset(Dataset):
    """Return (image, label) pairs where images are uniform random noise."""

    def __init__(self, length: int, image_size: int):
        self.len = length
        self.image_size = image_size

    def __len__(self):  # noqa: D401
        return self.len

    def __getitem__(self, idx):  # noqa: D401,ARG002
        img = torch.rand(3, self.image_size, self.image_size) * 2 - 1  # [-1,1]
        label = 0  # dummy
        return img, label


def get_dataloaders(batch_size: int, image_size: int) -> Tuple[DataLoader, DataLoader]:  # noqa: D401
    train_ds = RandomImageDataset(1024, image_size)
    val_ds = RandomImageDataset(256, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
