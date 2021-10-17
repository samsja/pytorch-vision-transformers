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
        dim=128,
        depth=12,
        patch_size=4,
        kernel_size=7,
    )


class ConvMixerModule(TransformersModule):
    def __init__(self, num_classes: int, lr=1e-3):
        super().__init__(TinyConvMixer(num_classes), lr)
