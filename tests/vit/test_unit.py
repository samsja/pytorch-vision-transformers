import pytest
import torch

from vision_transformers.vit import (
    EmbeddedPatch,
    Encoder,
    EncoderBlock,
    MHSAttention,
    PatchSizeError,
    positional_embedding,
)


def test_patch_size_error():
    try:
        em_patch = EmbeddedPatch(3, (28, 28), dim=512)
    except PatchSizeError:
        assert True
    else:
        assert False


def test_embedded_shape():

    input_ = torch.zeros((2, 3, 28, 28))
    em_patch = EmbeddedPatch(14, (28, 28), dim=512)
    patches = em_patch(input_)

    assert patches.shape == (2, 4, 512)


def test_positional_em():

    assert positional_embedding(512).shape == (512,)


def test_SHA():
    attention = MHSAttention(512, 1)

    assert attention(torch.zeros(2, 4, 512)).shape == (2, 4, 512)


def test_MSHA():

    attention = MHSAttention(512, 8)
    assert attention(torch.zeros(2, 4, 512)).shape == (2, 4, 512)


def test_encoder_block():

    encoder = EncoderBlock(512)
    patch = torch.zeros(2, 4, 512)

    assert encoder(patch).shape == patch.shape


def test_encoder():

    encoder = Encoder(512, 2)
    patch = torch.zeros(2, 4, 512)

    assert encoder(patch).shape == patch.shape
