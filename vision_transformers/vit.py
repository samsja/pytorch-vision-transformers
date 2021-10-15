import math
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
        self,
        size_of_patch: int,
        input_shape: Tuple[int, int],
        dim,
        channels=3,
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


class MHSAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()

        self.heads = num_heads

        if dim % num_heads != 0:
            raise DimError(
                f" input dim {dim} is not compatible with {num_heads} attention heads "
            )

        self.dim_heads = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * self.dim_heads * self.heads)
        self.linear_out = (
            nn.Linear(self.dim_heads * self.heads, dim)
            if self.heads > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q, k, v = list(
            map(
                lambda x: rearrange(
                    x, "b p (heads d) -> b heads p d", heads=self.heads
                ),
                self.qkv(x).chunk(3, dim=-1),
            )
        )

        A = q @ k.transpose(-1, -2) / math.sqrt(self.dim_heads)
        MSA = (A.softmax(dim=2)) @ v
        MSA = rearrange(MSA, "b heads p d  -> b p (heads d)")
        SA = self.linear_out(MSA)

        return SA
