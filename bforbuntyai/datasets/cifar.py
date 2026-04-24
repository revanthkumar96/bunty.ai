import numpy as np

from ._base import BaseDataset
from .._utils import get_cache_dir


class CIFAR10(BaseDataset):
    """CIFAR-10 colour images (32×32 RGB, 10 classes)."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        x_train, y_train, x_test, y_test = self._load()
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.name = "CIFAR10"
        self.shape = (32, 32, 3)
        self.num_classes = 10
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    def _load(self):
        try:
            from torchvision import datasets as tvds

            cache = get_cache_dir()
            train_ds = tvds.CIFAR10(root=str(cache), train=True, download=True)
            test_ds = tvds.CIFAR10(root=str(cache), train=False, download=True)
            x_train = np.array(train_ds.data, dtype="float32") / 255.0   # (N,32,32,3)
            y_train = np.array(train_ds.targets)
            x_test = np.array(test_ds.data, dtype="float32") / 255.0
            y_test = np.array(test_ds.targets)
            return x_train, y_train, x_test, y_test
        except ImportError:
            pass

        try:
            from tensorflow.keras.datasets import cifar10

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            return x_train, y_train, x_test, y_test
        except ImportError:
            pass

        raise ImportError(
            "CIFAR10 requires torch+torchvision or tensorflow.\n"
            "Install with: pip install bforbuntyai[torch]"
        )
