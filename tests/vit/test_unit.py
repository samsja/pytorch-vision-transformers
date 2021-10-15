import pytest
import torch

from vision_transformers.vit import EmbeddedPatch, positional_embedding


@pytest.mark.xfail
def test_patch_size_error():
    em_patch = EmbeddedPatch(3, (28, 28))


def test_embedded_shape():

    input_ = torch.zeros((2, 3, 28, 28))

    em_patch = EmbeddedPatch(14, (28, 28), dim=512)

    patches = em_patch(input_)

    assert patches.shape == (2, 4, 512)


def test_positional_em():

    assert positional_embedding(512).shape == (512,)
