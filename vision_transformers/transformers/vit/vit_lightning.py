import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from vision_transformers.transformers.vit.vit import ViT


def TinyViT(num_classes: int) -> ViT:
    return ViT(
        num_classes=10,
        dim=128,
        length=12,
        heads=8,
        size_of_patch=4,
        input_shape=(28, 28),
    )


class ViTModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        print(self.device)
        self.vit = TinyViT(num_classes)
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()

    def forward(self, x):
        return self.vit(x)

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
