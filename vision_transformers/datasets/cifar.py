from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMAGE_SHAPE = (28, 28)


def get_transforms(image_shape: Tuple[int, int]) -> Callable:

    all_transform = [
        torchvision.transforms.RandomCrop(_IMAGE_SHAPE),
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
    ]

    return torchvision.transforms.Compose(all_transform)


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
            dataset = CIFAR10(
                self.data_path, transform=get_transforms(_IMAGE_SHAPE), train=True
            )
        except RuntimeError:
            dataset = CIFAR10(
                self.data_path,
                transform=get_transforms(_IMAGE_SHAPE),
                train=True,
                download=True,
            )

        val_len = int(_VAL_PERCENTAGE * len(dataset))

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [
                len(dataset) - val_len,
                val_len,
            ],
            generator=torch.Generator().manual_seed(42),
        )
        try:
            self.test_dataset = CIFAR10(
                self.data_path, transform=get_transforms(_IMAGE_SHAPE), train=False
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
