from typing import Optional

import numpy as np

from ._base import BaseDataset
from .._utils import get_cache_dir


def _load_via_torchvision(name: str):
    from torchvision import datasets as tvds

    cache = get_cache_dir()
    cls = tvds.MNIST if name == "MNIST" else tvds.FashionMNIST
    train_ds = cls(root=str(cache), train=True, download=True)
    test_ds = cls(root=str(cache), train=False, download=True)
    x_train = train_ds.data.numpy().astype("float32") / 255.0
    y_train = train_ds.targets.numpy()
    x_test = test_ds.data.numpy().astype("float32") / 255.0
    y_test = test_ds.targets.numpy()
    return x_train, y_train, x_test, y_test


def _load_via_keras(name: str):
    if name == "MNIST":
        from tensorflow.keras.datasets import mnist as ds
    else:
        from tensorflow.keras.datasets import fashion_mnist as ds

    (x_train, y_train), (x_test, y_test) = ds.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def _build(name: str, batch_size: int):
    try:
        x_train, y_train, x_test, y_test = _load_via_torchvision(name)
    except ImportError:
        try:
            x_train, y_train, x_test, y_test = _load_via_keras(name)
        except ImportError:
            raise ImportError(
                f"{name} requires either torch+torchvision or tensorflow.\n"
                "Install with: pip install bforbuntyai[torch]  "
                "or  pip install bforbuntyai[tensorflow]"
            )

    # Ensure channel dim: (N, H, W) → (N, H, W, 1)
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test


class MNIST(BaseDataset):
    """MNIST handwritten digits (28×28 grayscale, 10 classes)."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        x_train, y_train, x_test, y_test = _build("MNIST", batch_size)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.name = "MNIST"
        self.shape = (28, 28, 1)
        self.num_classes = 10
        self.class_names = [str(i) for i in range(10)]


class FashionMNIST(BaseDataset):
    """Fashion-MNIST clothing items (28×28 grayscale, 10 classes)."""

    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        x_train, y_train, x_test, y_test = _build("FashionMNIST", batch_size)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.name = "FashionMNIST"
        self.shape = (28, 28, 1)
        self.num_classes = 10
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
