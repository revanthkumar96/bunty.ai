"""GAN / DCGAN / ConditionalGAN unit tests."""

import numpy as np
import pytest


def _fake_ds(h=28, w=28, c=1, n=200):
    from bforbuntyai.datasets._base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.name = "Fake"
    ds.shape = (h, w, c)
    ds.num_classes = 10
    ds.class_names = [str(i) for i in range(10)]
    ds.batch_size = 32
    rng = np.random.default_rng(2)
    ds.x_train = rng.random((n, h, w, c), dtype=np.float32)
    ds.y_train = rng.integers(0, 10, n)
    ds.x_test = rng.random((20, h, w, c), dtype=np.float32)
    ds.y_test = rng.integers(0, 10, 20)
    return ds


def test_gan_train_and_generate():
    pytest.importorskip("torch")
    from bforbuntyai import GAN

    ds = _fake_ds()
    gan = GAN(ds, latent_dim=16, batch_size=32)
    result = gan.train(epochs=2)
    assert result is gan
    assert len(gan.g_losses) == 2

    imgs = gan.generate(n=9, return_array=True)
    assert imgs.shape == (9, 28, 28, 1)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0


def test_gan_save_load(tmp_path):
    pytest.importorskip("torch")
    from bforbuntyai import GAN

    ds = _fake_ds()
    gan = GAN(ds, latent_dim=16, batch_size=32)
    gan.train(epochs=1)
    path = str(tmp_path / "gan.pth")
    gan.save(path)
    gan.load(path)
    imgs = gan.generate(n=4, return_array=True)
    assert imgs.shape[0] == 4


def test_dcgan_train_and_generate():
    pytest.importorskip("torch")
    from bforbuntyai import DCGAN

    ds = _fake_ds()
    dcgan = DCGAN(ds, latent_dim=16, batch_size=32)
    dcgan.train(epochs=2)
    imgs = dcgan.generate(n=4, return_array=True)
    assert imgs.shape == (4, 28, 28, 1)


def test_conditional_gan_train():
    pytest.importorskip("torch")
    from bforbuntyai import ConditionalGAN

    ds = _fake_ds()
    cgan = ConditionalGAN(ds, num_classes=10, latent_dim=16, batch_size=32)
    cgan.train(epochs=2)
    imgs = cgan.generate(n=5, return_array=True)
    assert imgs.shape == (5, 28, 28, 1)


def test_conditional_gan_generate_class():
    pytest.importorskip("torch")
    from bforbuntyai import ConditionalGAN
    import matplotlib
    matplotlib.use("Agg")

    ds = _fake_ds()
    cgan = ConditionalGAN(ds, num_classes=10, latent_dim=16, batch_size=32)
    cgan.train(epochs=1)
    imgs = cgan.generate_class(labels=[0, 1, 2])
    assert imgs.shape[0] == 3
