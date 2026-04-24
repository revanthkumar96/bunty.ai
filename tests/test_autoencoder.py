"""AutoEncoder unit tests — uses a tiny fake dataset, no real data downloads."""

import numpy as np
import pytest


def _fake_ds(h=28, w=28, c=1):
    from bforbuntyai.datasets._base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.name = "Fake"
    ds.shape = (h, w, c)
    ds.num_classes = 10
    ds.class_names = [str(i) for i in range(10)]
    ds.batch_size = 32
    rng = np.random.default_rng(1)
    ds.x_train = rng.random((200, h, w, c), dtype=np.float32)
    ds.y_train = rng.integers(0, 10, 200)
    ds.x_test = rng.random((50, h, w, c), dtype=np.float32)
    ds.y_test = rng.integers(0, 10, 50)
    return ds


def test_autoencoder_train_mnist(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from bforbuntyai import AutoEncoder

    ds = _fake_ds(28, 28, 1)
    ae = AutoEncoder(ds, encoding_dim=16)
    result = ae.train(epochs=2, batch_size=64, validation_split=0.1)
    assert result is ae, "train() should return self"
    assert ae.model is not None


def test_autoencoder_generate():
    pytest.importorskip("tensorflow")
    from bforbuntyai import AutoEncoder

    ds = _fake_ds()
    ae = AutoEncoder(ds, encoding_dim=16)
    ae.train(epochs=1, batch_size=64)
    out = ae.generate(n=5)
    assert out.shape == (5, 28, 28, 1)


def test_autoencoder_save_load(tmp_path):
    pytest.importorskip("tensorflow")
    from bforbuntyai import AutoEncoder

    ds = _fake_ds()
    ae = AutoEncoder(ds, encoding_dim=8)
    ae.train(epochs=1, batch_size=64)

    save_path = str(tmp_path / "ae.keras")
    ae.save(save_path)

    ae2 = AutoEncoder(ds)
    ae2.load(save_path)
    out = ae2.generate(n=3)
    assert out.shape[0] == 3


def test_autoencoder_cifar(tmp_path):
    pytest.importorskip("tensorflow")
    from bforbuntyai import AutoEncoder

    ds = _fake_ds(32, 32, 3)
    ae = AutoEncoder(ds, encoding_dim=32, loss="mse")
    ae.train(epochs=1, batch_size=64)
    out = ae.generate(n=4)
    assert out.shape == (4, 32, 32, 3)
