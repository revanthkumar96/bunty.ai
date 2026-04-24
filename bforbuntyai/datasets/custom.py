from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ._base import BaseDataset

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class Custom(BaseDataset):
    """Load images from a local directory.

    Directory layouts supported:
        flat/        — all images in one folder (single class)
        labeled/
            class_a/ — one sub-folder per class
            class_b/
        split/
            train/
            test/

    Usage:
        data = Custom("path/to/images", image_size=64)
        data = Custom("path/to/labeled", image_size=128)
    """

    def __init__(
        self,
        path: str,
        image_size: int = 64,
        batch_size: int = 64,
        max_samples: Optional[int] = None,
    ):
        self.path = Path(path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._load()

    def _load(self) -> None:
        train_dir = self.path / "train"
        test_dir = self.path / "test"

        if train_dir.exists():
            x_train, y_train, names = self._load_dir(train_dir)
            if test_dir.exists():
                x_test, y_test, _ = self._load_dir(test_dir)
            else:
                split = int(len(x_train) * 0.9)
                x_test, y_test = x_train[split:], y_train[split:]
                x_train, y_train = x_train[:split], y_train[:split]
        else:
            x_all, y_all, names = self._load_dir(self.path)
            split = int(len(x_all) * 0.9)
            x_train, y_train = x_all[:split], y_all[:split]
            x_test, y_test = x_all[split:], y_all[split:]

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.class_names = names
        self.num_classes = len(names)
        self.shape = (self.image_size, self.image_size, 3)
        self.name = "Custom"

    def _load_dir(self, directory: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        from PIL import Image

        subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])

        if subdirs:
            class_names = [d.name for d in subdirs]
            images, labels = [], []
            for label, subdir in enumerate(subdirs):
                files = [f for f in subdir.iterdir() if f.suffix.lower() in _IMG_EXTS]
                if self.max_samples:
                    files = files[: self.max_samples // len(subdirs)]
                for f in files:
                    img = (
                        Image.open(f)
                        .convert("RGB")
                        .resize((self.image_size, self.image_size))
                    )
                    images.append(np.array(img, dtype="float32") / 255.0)
                    labels.append(label)
        else:
            class_names = ["images"]
            files = [f for f in directory.iterdir() if f.suffix.lower() in _IMG_EXTS]
            if self.max_samples:
                files = files[: self.max_samples]
            images = [
                np.array(
                    Image.open(f).convert("RGB").resize((self.image_size, self.image_size)),
                    dtype="float32",
                )
                / 255.0
                for f in files
            ]
            labels = [0] * len(images)

        x = np.stack(images) if images else np.zeros((0, self.image_size, self.image_size, 3))
        y = np.array(labels)
        return x, y, class_names
