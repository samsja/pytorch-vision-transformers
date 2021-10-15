import pytest
import torch

from vision_transformers.vit import (
    EmbeddedPatch,
    Encoder,
    EncoderBlock,
    MHSAttention,
    PatchSizeError,
    Transformers,
    ViT,
    positional_embedding,
)


@torch.no_grad()
def test_vit_shape():

    model = ViT(
        num_classes=10,
        dim=512,
        length=2,
        heads=8,
        size_of_patch=14,
        input_shape=(28, 28),
    )

    imgs = torch.zeros(2, 3, 28, 28)

    assert model(imgs).shape == (2, 10)


@torch.no_grad()
def test_patch_size_error():
    try:
        em_patch = EmbeddedPatch(3, (28, 28), dim=512)
    except PatchSizeError:
        assert True
    else:
        assert False


@torch.no_grad()
def test_embedded_shape():

    input_ = torch.zeros((2, 3, 28, 28))
    em_patch = EmbeddedPatch(14, (28, 28), dim=512)
    patches = em_patch(input_)

    assert patches.shape == (2, 4, 512)


@torch.no_grad()
def test_embedded_number_of_patch():

    input_ = torch.zeros((2, 3, 28, 28))
    em_patch = EmbeddedPatch(14, (28, 28), dim=512)
    patches = em_patch(input_)

    assert em_patch.number_of_patch == 4
    assert patches.shape[1] == em_patch.number_of_patch


@torch.no_grad()
def test_positional_em():

    assert positional_embedding(512).shape == (512,)


@torch.no_grad()
def test_SHA():
    attention = MHSAttention(512, 1)

    assert attention(torch.zeros(2, 4, 512)).shape == (2, 4, 512)


@torch.no_grad()
def test_MSHA():

    attention = MHSAttention(512, 8)
    assert attention(torch.zeros(2, 4, 512)).shape == (2, 4, 512)


@torch.no_grad()
def test_encoder_block():

    encoder = EncoderBlock(512)
    patch = torch.zeros(2, 4, 512)

    assert encoder(patch).shape == patch.shape


@torch.no_grad()
def test_encoder():

    encoder = Encoder(512, 2)
    patch = torch.zeros(2, 4, 512)

    assert encoder(patch).shape == patch.shape


@torch.no_grad()
def test_transformers():

    transformers = Transformers(10, 4, 512, 2, 8)
    patch = torch.zeros(2, 4, 512)

    assert transformers(patch).shape == (2, 10)
