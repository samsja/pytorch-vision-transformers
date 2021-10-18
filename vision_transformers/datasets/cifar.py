import copy
from typing import Callable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_IMAGE_SHAPE_VAL = (32, 32)
_IMAGE_SHAPE = (30, 30)


def get_transforms(image_shape: Tuple[int, int]) -> Callable:

    tfm = rand_augment_transform(
        config_str="rand-m9-mstd0.5",
        hparams={"translate_const": 117, "img_mean": (124, 116, 104)},
    )

    resize = RandomResizedCropAndInterpolation(size=image_shape)

    all_transform = [
        torchvision.transforms.Lambda(lambda x: resize(x)),
        torchvision.transforms.Lambda(lambda x: tfm(x)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=_MEAN, std=_STD),
        torchvision.transforms.RandomErasing(
            p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=True
        ),
    ]

    return torchvision.transforms.Compose(all_transform)


def get_transforms_val(image_shape: Tuple[int, int]) -> Callable:

    all_transform = [
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
        self.data_module.train_dataset.dataset.transform = get_transforms((size, size))
