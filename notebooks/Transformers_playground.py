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
# -

# +
import os

from einops import rearrange

os.chdir("..")
# -

from vision_transformers.cifar import CIFARDataModule
from vision_transformers.resnet import ResNet
from vision_transformers.utils_plot import imshow
from vision_transformers.vit import EmbeddedPatch, ViT

# ## Dataset

data = CIFARDataModule("data", 10, 0)

data.setup()

imshow(data.train_dataset[0][0])

batch = torch.stack([data.train_dataset[0][0], data.train_dataset[1][0]])
batch.shape

model = ViT(
    num_classes=10, dim=128, length=12, heads=8, size_of_patch=4, input_shape=(28, 28)
)
