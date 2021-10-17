import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models import resnet18


class ResNet(torch.nn.Module):
    def __init__(self, num_classes: int, pretrained=True):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ResNetModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.resnet = ResNet(num_classes, pretrained=False)
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        acc = self.acc_fn(output, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
