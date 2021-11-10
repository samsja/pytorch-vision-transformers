import matplotlib.pyplot as plt
import numpy as np
import torch

from vision_transformers.data.utils.augmentation import _MEAN, _STD


def img_from_tensor(inp):
    inp = inp.to("cpu").numpy().transpose((1, 2, 0))
    mean = np.array(_MEAN)
    std = np.array(_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = img_from_tensor(inp)
    plt.imshow(inp)
    plt.axis("off")
    if title is not None:
        plt.title(title)
