from typing import Any, List, Optional, Tuple

import numpy as np


class BaseDataset:
    """Base class for all bforbuntyai datasets.

    Subclasses store data as numpy arrays (N, H, W, C) in [0, 1] for images.
    Use as_numpy() or as_torch_loader() to access data in the format your model needs.
    """

    name: str = ""
    shape: Tuple[int, ...] = ()   # (H, W, C) for image datasets
    num_classes: int = 0
    class_names: List[str] = []

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def as_numpy(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) as numpy arrays. X is float32 in [0, 1]."""
        if split == "train":
            return self.x_train, self.y_train
        return self.x_test, self.y_test

    def as_torch_loader(
        self,
        split: str = "train",
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        gan: bool = False,
    ) -> Any:
        """Return a PyTorch DataLoader.

        Args:
            split:      'train' or 'test'
            batch_size: overrides dataset default
            shuffle:    defaults to True for train, False for test
            gan:        if True, normalise images to [-1, 1] (required for Tanh output models)
        """
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch is required for as_torch_loader().\n"
                "Install with: pip install bforbuntyai[torch]"
            )

        bs = batch_size or getattr(self, "batch_size", 64)
        sh = (split == "train") if shuffle is None else shuffle

        if split == "train":
            x, y = self.x_train, self.y_train
        else:
            x, y = self.x_test, self.y_test

        # (N, H, W, C) → (N, C, H, W)
        x_t = torch.FloatTensor(np.transpose(x, (0, 3, 1, 2)))
        if gan:
            x_t = x_t * 2.0 - 1.0  # [0,1] → [-1,1]
        y_t = torch.LongTensor(y)

        return DataLoader(TensorDataset(x_t, y_t), batch_size=bs, shuffle=sh)

    def __len__(self) -> int:
        return len(self.x_train)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"train={len(self.x_train)}, test={len(self.x_test)}, "
            f"shape={self.shape}, classes={self.num_classes})"
        )
