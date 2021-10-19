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
from vision_transformers.datasets.utils.utils_plot import imshow
from vision_transformers.transformers.conv_mixer.conv_mixer import (
    ConvMixer,
    ConvMixerLayer,
    EmbeddingPatch,
)
from vision_transformers.transformers.vit.vit_lightning import ViTModule

# ## Dataset

data = CIFARDataModule("data", 2, 0)

data.setup()

imshow(data.train_dataset[0][0])
