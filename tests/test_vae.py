"""VAE unit tests."""

import numpy as np
import pytest


def _fake_ds(latent=2, n=200):
    from bforbuntyai.datasets._base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.name = "Fake"
    ds.shape = (28, 28, 1)
    ds.num_classes = 10
    ds.class_names = [str(i) for i in range(10)]
    ds.batch_size = 32
    rng = np.random.default_rng(3)
    ds.x_train = rng.random((n, 28, 28, 1), dtype=np.float32)
    ds.y_train = rng.integers(0, 10, n)
    ds.x_test = rng.random((50, 28, 28, 1), dtype=np.float32)
    ds.y_test = rng.integers(0, 10, 50)
    return ds


def test_vae_train():
    pytest.importorskip("torch")
    from bforbuntyai import VAE

    ds = _fake_ds()
    vae = VAE(ds, latent_dim=2, hidden_dim=64, batch_size=32)
    result = vae.train(epochs=2)
    assert result is vae
    assert len(vae.losses) == 2


def test_vae_generate():
    pytest.importorskip("torch")
    from bforbuntyai import VAE

    ds = _fake_ds()
    vae = VAE(ds, latent_dim=2, hidden_dim=64, batch_size=32)
    vae.train(epochs=1)
    imgs = vae.generate(n=8, return_array=True)
    assert imgs.shape == (8, 28, 28, 1)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0


def test_vae_interpolate():
    pytest.importorskip("torch")
    from bforbuntyai import VAE
    import matplotlib
    matplotlib.use("Agg")

    ds = _fake_ds()
    vae = VAE(ds, latent_dim=2, hidden_dim=64, batch_size=32)
    vae.train(epochs=1)
    img_a = ds.x_test[0]
    img_b = ds.x_test[1]
    result = vae.interpolate(img_a, img_b, steps=5)
    assert result.shape == (5, 28, 28, 1)


def test_vae_save_load(tmp_path):
    pytest.importorskip("torch")
    from bforbuntyai import VAE

    ds = _fake_ds()
    vae = VAE(ds, latent_dim=2, hidden_dim=64, batch_size=32)
    vae.train(epochs=1)
    path = str(tmp_path / "vae.pth")
    vae.save(path)
    vae.load(path)
    imgs = vae.generate(n=4, return_array=True)
    assert imgs.shape[0] == 4
