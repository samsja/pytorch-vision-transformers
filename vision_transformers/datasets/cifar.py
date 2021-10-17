from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_IMAGE_SHAPE = (30, 30)


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
            self.dataset = CIFAR10(
                self.data_path, transform=get_transforms(_IMAGE_SHAPE), train=True
            )
        except RuntimeError:
            self.dataset = CIFAR10(
                self.data_path,
                transform=get_transforms(_IMAGE_SHAPE),
                train=True,
                download=True,
            )

        val_len = int(_VAL_PERCENTAGE * len(self.dataset))

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                len(self.dataset) - val_len,
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


class ProgressiveImageResizing(Callback):
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        epoch_final_size: int,
        n_step: int,
        init_size: int,
        final_size: int,
    ):
        super().__init__()
        self.data_module = data_module

        step = epoch_final_size // n_step
        self.epoch_to_change = np.linspace(
            1, epoch_final_size, n_step, dtype=int
        ).tolist()
        self.new_size = np.linspace(init_size, final_size, n_step, dtype=int).tolist()

        self.epoch = 1

        self._change_size(self.new_size[0])

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self.epoch += 1

        if self.epoch in self.epoch_to_change:
            idx = self.epoch_to_change.index(self.epoch)
            self._change_size(self.new_size[idx])
            self.log("size", self.new_size[idx])

    def _change_size(self, size: int):
        self.data_module.dataset.transform = get_transforms((size, size))
