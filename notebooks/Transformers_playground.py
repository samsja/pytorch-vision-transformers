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
import torch
import torch.nn as nn

torch.__version__

# +
import os

from einops import rearrange

os.chdir("..")
# -

from vision_transformers.datasets.cifar import CIFARDataModule
from vision_transformers.transformers.conv_mixer.conv_mixer import (
    ConvMixer,
    ConvMixerLayer,
)
from vision_transformers.transformers.vit.vit_lightning import ViTModule
from vision_transformers.utils.utils_plot import imshow

# ## Dataset

data = CIFARDataModule("data", 10, 0)

data.setup()

imshow(data.train_dataset[0][0])

batch = torch.stack([data.train_dataset[0][0], data.train_dataset[1][0]])
batch.shape

model = ConvMixer(64, 1, 7, 4, 10)

model(batch).shape
