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

import pytorch_lightning as pl

# +
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.__version__

# +
import os

os.chdir("..")
# -

from vision_transformers.datasets.cifar import CIFARDataModule, ProgressiveImageResizing
from vision_transformers.datasets.utils.utils_plot import imshow
from vision_transformers.transformers.conv_mixer.conv_mixer_lightning import (
    ConvMixerModule,
)
from vision_transformers.transformers.vit.vit_lightning import ViTModule

# ## Dataset

## PARAMS
batch_size = 128
num_workers = 8
patience = 4
model_path = "data/models/conv_mixer"
epochs = 100

data = CIFARDataModule("data", batch_size, num_workers)

data.setup()

imshow(data.train_dataset[0][0])

# ## Model

model = ConvMixerModule(10, 1e-2)

# + [markdown] tags=[]
#
# ## Training
# -

increase_image_shape = ProgressiveImageResizing(
    data, epoch_final_size=15, n_step=15, init_size=5, final_size=30
)

increase_image_shape.epoch_to_change, increase_image_shape.new_size

callbacks = [
    EarlyStopping(monitor="val_acc", mode="max", patience=patience, strict=False),
    increase_image_shape,
]

trainer = pl.Trainer(
    gpus=1,
    max_epochs=epochs,
    check_val_every_n_epoch=1,
    log_every_n_steps=3,
    callbacks=callbacks,
    gradient_clip_val=1.0,
    #  precision=16,
)

trainer.fit(model, data)
