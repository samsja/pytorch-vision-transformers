import torch

from vision_transformers.datasets.cifar import _IMAGE_SHAPE
from vision_transformers.transformers.resnet import ResNet


def test_resnet_output_shape():
    model = ResNet(10, pretrained=False)

    output = model(torch.zeros((2, 3, *_IMAGE_SHAPE)))

    assert output.shape == (2, 10)
