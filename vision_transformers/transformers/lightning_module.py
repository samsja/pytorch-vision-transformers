from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR


class TransformersModule(pl.LightningModule):
    def __init__(
        self, model: nn.Module, lr=1e-3, one_cycle_scheduler: Dict[str, int] = None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_fn = torchmetrics.Accuracy()
        self.one_cycle_scheduler = one_cycle_scheduler

    def forward(self, x):
        return self.model(x)

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        acc = self.acc_fn(output, y)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.one_cycle_scheduler:
            lr_scheduler = OneCycleLR(optimizer, **self.one_cycle_scheduler)

            return [optimizer], [lr_scheduler]
        else:
            return optimizer
