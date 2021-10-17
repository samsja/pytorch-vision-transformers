# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# ## Import

# +
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# -

torch.__version__

import os

os.chdir("..")

from vision_transformers.datasets.cifar import CIFARDataModule
from vision_transformers.transformers.vit.vit_lightning import ViTModule
from vision_transformers.utils.utils_plot import imshow

# ## Dataset

## PARAMS
batch_size = 256
num_workers = 8
patience = 10
model_path = "data/models/vit"
epochs = 200

data = CIFARDataModule("data", batch_size, num_workers)

data.setup()

imshow(data.train_dataset[0][0])

model = ViTModule(10, 1e-3)

callbacks = [
    # EarlyStopping(
    #    monitor="val_acc",
    #    stopping_threshold=0.97,
    #    patience=patience,
    #    min_delta=0.0,
    # ),
    ModelCheckpoint(
        dirpath=model_path,
        monitor="val_acc",
        save_top_k=3,
        filename="{epoch}-{val_abs:.2f}-{val_loss:.2f}",
        save_last=True,
    ),
]

trainer = pl.Trainer(
    gpus=1,
    max_epochs=epochs,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    callbacks=callbacks,
    #  precision=16,
)

trainer.fit(model, data)
