"""Dataset layer unit tests — no GPU or large downloads required."""

import numpy as np
import pytest


def _make_fake_dataset(h=28, w=28, c=1, n_train=100, n_test=20, num_classes=10):
    from bforbuntyai.datasets._base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.name = "Fake"
    ds.shape = (h, w, c)
    ds.num_classes = num_classes
    ds.class_names = [str(i) for i in range(num_classes)]
    ds.batch_size = 32
    rng = np.random.default_rng(0)
    ds.x_train = rng.random((n_train, h, w, c), dtype=np.float32)
    ds.y_train = rng.integers(0, num_classes, n_train)
    ds.x_test = rng.random((n_test, h, w, c), dtype=np.float32)
    ds.y_test = rng.integers(0, num_classes, n_test)
    return ds


def test_base_as_numpy():
    ds = _make_fake_dataset()
    x, y = ds.as_numpy("train")
    assert x.shape == (100, 28, 28, 1)
    assert y.shape == (100,)
    assert x.max() <= 1.0 and x.min() >= 0.0


def test_base_as_numpy_test_split():
    ds = _make_fake_dataset()
    x, y = ds.as_numpy("test")
    assert x.shape == (20, 28, 28, 1)


def test_base_as_torch_loader():
    pytest.importorskip("torch")
    ds = _make_fake_dataset()
    loader = ds.as_torch_loader(split="train", batch_size=16)
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape == (16, 1, 28, 28), "Should be (B, C, H, W)"
    assert float(batch_imgs.min()) >= 0.0
    assert float(batch_imgs.max()) <= 1.0


def test_base_as_torch_loader_gan_normalize():
    pytest.importorskip("torch")
    ds = _make_fake_dataset()
    loader = ds.as_torch_loader(split="train", batch_size=16, gan=True)
    batch_imgs, _ = next(iter(loader))
    assert float(batch_imgs.min()) >= -1.0
    assert float(batch_imgs.max()) <= 1.0


def test_base_len():
    ds = _make_fake_dataset(n_train=100)
    assert len(ds) == 100


def test_base_repr():
    ds = _make_fake_dataset()
    r = repr(ds)
    assert "Fake" in r


def test_custom_dataset(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "class_a"
    img_dir.mkdir()
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(img_dir / f"img_{i}.png")

    img_dir2 = tmp_path / "class_b"
    img_dir2.mkdir()
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(img_dir2 / f"img_{i}.png")

    from bforbuntyai.datasets.custom import Custom

    ds = Custom(str(tmp_path), image_size=16)
    assert ds.num_classes == 2
    assert ds.shape == (16, 16, 3)
    assert len(ds) > 0
