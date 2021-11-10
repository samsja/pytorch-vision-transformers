import copy
from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from vision_transformers.data.utils.augmentation import (
    get_transforms,
    get_transforms_val,
)

_IMAGE_SHAPE_VAL = (32, 32)
_IMAGE_SHAPE = (30, 30)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        _VAL_PERCENTAGE = 0.1

        try:
            self.train_dataset = CIFAR10(
                self.data_path, transform=get_transforms(_IMAGE_SHAPE), train=True
            )
        except RuntimeError:
            self.train_dataset = CIFAR10(
                self.data_path,
                transform=get_transforms(_IMAGE_SHAPE),
                train=True,
                download=True,
            )

        self.val_dataset_base = CIFAR10(
            self.data_path, transform=get_transforms_val(_IMAGE_SHAPE_VAL), train=True
        )

        val_len = int(_VAL_PERCENTAGE * len(self.train_dataset))

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [
                len(self.train_dataset) - val_len,
                val_len,
            ],
            generator=torch.Generator().manual_seed(42),
        )

        self.val_dataset.dataset = self.val_dataset_base

        try:
            self.test_dataset = CIFAR10(
                self.data_path,
                transform=get_transforms_val(_IMAGE_SHAPE_VAL),
                train=False,
            )
        except RuntimeError:
            self.test_dataset = CIFAR10(
                self.data_path,
                transform=get_transforms(_IMAGE_SHAPE),
                train=False,
                download=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
