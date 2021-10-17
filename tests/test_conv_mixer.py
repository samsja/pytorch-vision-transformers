import torch

from vision_transformers.transformers.conv_mixer.conv_mixer import (
    ConvMixer,
    ConvMixerLayer,
)


def test_shape_conv_mixer():

    layer = ConvMixer(64, 2, 7, 4, 10)
    input_ = torch.zeros(2, 3, 28, 28)

    assert layer(input_).shape == (2, 10)


def test_shape_conv_block():

    layer = ConvMixerLayer(64, 7)
    input_ = torch.zeros(2, 64, 4, 4)

    assert layer(input_).shape == input_.shape
