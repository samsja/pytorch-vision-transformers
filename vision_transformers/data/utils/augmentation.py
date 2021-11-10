from typing import Callable, Tuple

import torch
import torchvision
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_transforms(image_shape: Tuple[int, int]) -> Callable:

    tfm = rand_augment_transform(
        config_str="rand-m9-mstd0.5",
        hparams={},
    )

    resize = RandomResizedCropAndInterpolation(size=image_shape)

    all_transform = [
        torchvision.transforms.Lambda(lambda x: resize(x)),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.Lambda(lambda x: tfm(x))], p=0.7
        ),
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
