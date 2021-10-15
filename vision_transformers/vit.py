from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


class VitError(ValueError):
    pass


class PatchSizeError(VitError):
    pass


class EmbeddedPatch(nn.Module):
    def __init__(
        self, size_of_patch: int, input_shape: Tuple[int, int], channels=3, dim=512
    ):
        super().__init__()

        if (input_shape[0] % size_of_patch != 0) or (
            input_shape[1] % size_of_patch != 0
        ):
            raise PatchSizeError(
                f"size patch of {size_of_patch} is not compatible with image shape {input_shape}"
            )

        self.P = size_of_patch
        self.dim = dim

        self.linear = nn.Linear(channels * self.P * self.P, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor():
        x = rearrange(
            x, " n c (h p1) (w p2) -> n (h w) (p1 p2 c)", p2=self.P, p1=self.P
        )
        x = self.linear(x)
        pos_em = positional_embedding(self.dim)
        return x + pos_em


class DimError(VitError):
    pass


def positional_embedding(dim: int):
    if dim % 2 != 0:
        raise DimError(f"dim {dim} should pair")

    d = dim // 2

    w = (10_000 * torch.ones(d)).pow(2 * torch.arange(d) / d)
    cos = torch.cos(w)
    sin = torch.sin(w)

    return rearrange([sin, cos], " a b -> (b a) ")
