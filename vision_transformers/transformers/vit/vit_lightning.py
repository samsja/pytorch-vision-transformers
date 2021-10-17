import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from vision_transformers.transformers.lightning_module import TransformersModule
from vision_transformers.transformers.vit.vit import ViT


def TinyViT(num_classes: int) -> ViT:
    return ViT(
        num_classes=10,
        dim=128,
        depth=12,
        heads=8,
        patch_size=4,
        input_shape=(28, 28),
    )


class ViTModule(TransformersModule):
    def __init__(self, num_classes: int, lr=1e-3):
        super().__init__(TinyViT(num_classes), lr)
