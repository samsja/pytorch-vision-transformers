import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from vision_transformers.transformers.conv_mixer.conv_mixer import ConvMixer
from vision_transformers.transformers.lightning_module import TransformersModule


def TinyConvMixer(num_classes: int) -> ConvMixer:
    return ConvMixer(
        num_classes=10,
        dim=256,
        depth=8,
        patch_size=2,
        kernel_size=3,
    )


class ConvMixerModule(TransformersModule):
    def __init__(self, num_classes: int, lr=1e-3):
        super().__init__(TinyConvMixer(num_classes), lr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
