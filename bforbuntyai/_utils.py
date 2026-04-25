from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import requests
from tqdm import tqdm

from ._logging import get_logger

CACHE_DIR = Path.home() / ".bforbuntyai" / "cache"
_logger = get_logger("utils")


def get_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _logger.info("Using device: %s", device)
        return device
    except ImportError:
        return None


def plot_grid(
    images: Sequence,
    titles: Optional[List[str]] = None,
    cols: int = 10,
    figsize=None,
    cmap: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    n = len(images)
    rows = max(1, (n + cols - 1) // cols)
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        ax = axes[i]
        img = np.array(img)
        img = np.clip(img, 0, 1) if img.dtype == np.float32 or img.max() <= 1.0 else img
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            ax.imshow(img.squeeze(), cmap=cmap or "gray")
        else:
            ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=8)
        ax.axis("off")

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def ensure_dir(path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
