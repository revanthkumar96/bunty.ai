from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .._logging import get_logger
from .._utils import download_file, get_cache_dir

_logger = get_logger("datasets.pix2pix")

EDGES2SHOES_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"


class PairedImageDataset:
    """PyTorch Dataset for paired image-to-image translation."""

    def __init__(self, pairs: np.ndarray, gan: bool = True):
        # pairs: (N, H, W*2, C) — left=edge, right=real
        self.pairs = pairs
        self.gan = gan

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple:
        import torch

        pair = self.pairs[idx]
        w = pair.shape[1] // 2
        edge = pair[:, :w, :]   # (H, W, C)
        real = pair[:, w:, :]

        # (H, W, C) → (C, H, W)
        edge_t = torch.FloatTensor(np.transpose(edge, (2, 0, 1)))
        real_t = torch.FloatTensor(np.transpose(real, (2, 0, 1)))

        if self.gan:
            edge_t = edge_t * 2.0 - 1.0
            real_t = real_t * 2.0 - 1.0

        return edge_t, real_t


class Edges2Shoes:
    """Edges→Shoes paired dataset for Pix2Pix.

    Downloads ~1.5 GB from Berkeley on first use and caches to ~/.bforbuntyai/cache.
    """

    def __init__(self, image_size: int = 256, batch_size: int = 16, max_samples: Optional[int] = None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.name = "Edges2Shoes"
        self.shape = (image_size, image_size, 3)
        self._load()

    def _load(self) -> None:
        from PIL import Image

        cache = get_cache_dir()
        archive = cache / "edges2shoes.tar.gz"
        data_dir = cache / "edges2shoes"

        if not data_dir.exists():
            _logger.info("Downloading Edges2Shoes dataset (~1.5 GB)...")
            download_file(EDGES2SHOES_URL, archive)
            import tarfile

            _logger.info("Extracting...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(cache)
            _logger.info("Done.")

        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

        self._train_pairs = self._load_pairs(train_dir)
        self._val_pairs = self._load_pairs(val_dir)

    def _load_pairs(self, directory: Path) -> np.ndarray:
        from PIL import Image

        pairs = []
        files = sorted(directory.glob("*.jpg"))
        if self.max_samples:
            files = files[: self.max_samples]

        for f in files:
            img = Image.open(f).convert("RGB").resize(
                (self.image_size * 2, self.image_size)
            )
            pairs.append(np.array(img, dtype="float32") / 255.0)

        return np.stack(pairs) if pairs else np.zeros((0, self.image_size, self.image_size * 2, 3))

    def as_torch_loader(self, split: str = "train", batch_size: Optional[int] = None, gan: bool = True):
        from torch.utils.data import DataLoader

        bs = batch_size or self.batch_size
        pairs = self._train_pairs if split == "train" else self._val_pairs
        ds = PairedImageDataset(pairs, gan=gan)
        return DataLoader(ds, batch_size=bs, shuffle=(split == "train"))

    def __len__(self) -> int:
        return len(self._train_pairs)

    def __repr__(self) -> str:
        return f"Edges2Shoes(train={len(self._train_pairs)}, val={len(self._val_pairs)}, size={self.image_size})"
