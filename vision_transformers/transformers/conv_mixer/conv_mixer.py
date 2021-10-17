from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


class ConvMixerError(ValueError):
    pass


class ConvMixer(nn.Module):
    def __init__(
        self, dim: int, depth: int, kernel_size: int, patch_size: int, num_classes: int
    ):
        super().__init__()

        self.dim = dim

        self.layers = nn.Sequential(
            *(ConvMixerLayer(dim, kernel_size) for _ in range(depth))
        )

        self.patch_em = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.patch_em(x)

        out = self.layers(out)

        return self.classifier(out)


class ConvMixerLayer(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()

        self.depthwise_conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.pointwise_conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.depthwise_conv_block(x) + x

        return self.pointwise_conv_block(out) + out
