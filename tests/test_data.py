from vision_transformers.datasets.cifar import CIFARDataModule


def test_CIRAR10_shape():
    data = CIFARDataModule("data", 1, 0)
    data.setup()

    assert data.train_dataset[0][0].shape == (3, 28, 28)
    assert data.val_dataset[0][0].shape == (3, 28, 28)
