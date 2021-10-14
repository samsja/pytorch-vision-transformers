from itertools import product
from typing import Tuple

import torch
import torch.nn as nn


class PatchSizeError(ValueError):
    pass


class EmbeddedPatch(nn.Module):
    def __init__(self, size_of_patch: int, input_shape: Tuple[int, int]):
        super().__init__()

        if (input_shape[0] % size_of_patch != 0) or (
            input_shape[1] % size_of_patch != 0
        ):
            raise PatchSizeError(
                f"size patch of {size_of_patch} is not compatible with image shape {input_shape}"
            )

        self.P = size_of_patch
        self.input_shape = input_shape
        self.N = (input_shape[0] // self.P) * (input_shape[1] // self.P)

    def forward(self, x: torch.Tensor) -> torch.Tensor():
        return torch.stack(
            [
                x[:, :, i * self.P : (i + 1) * self.P, j * self.P : (j + 1) * self.P]
                for i, j in product(
                    range(0, self.input_shape[0] // self.P),
                    range(0, self.input_shape[1] // self.P),
                )
            ],
            dim=1,
        )