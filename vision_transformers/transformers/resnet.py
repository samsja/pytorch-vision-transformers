import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models import resnet18

from vision_transformers.transformers.lightning_module import TransformersModule


class ResNet(torch.nn.Module):
    def __init__(self, num_classes: int, pretrained=True):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ResNetModule(TransformersModule):
    def __init__(self, num_classes: int, lr=1e-3, pretrained=True):
        super().__init__(ResNet(num_classes, pretrained), lr)
