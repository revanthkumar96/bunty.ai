"""TDD tests: structured logging and per-model metrics dict."""
import logging

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
    rng = np.random.default_rng(42)
    ds.x_train = rng.random((n, h, w, c), dtype=np.float32)
    ds.y_train = rng.integers(0, 10, n)
    ds.x_test = rng.random((20, h, w, c), dtype=np.float32)
    ds.y_test = rng.integers(0, 10, 20)
    return ds


# ─── Logging ────────────────────────────────────────────────────────────────

def test_get_logger_returns_namespaced_logger():
    from bforbuntyai._logging import get_logger

    logger = get_logger("models.gan")
    assert logger.name == "bforbuntyai.models.gan"
    assert isinstance(logger, logging.Logger)


def test_setup_logging_sets_level():
    from bforbuntyai import setup_logging

    setup_logging(level="DEBUG")
    root = logging.getLogger("bforbuntyai")
    assert root.level == logging.DEBUG
    setup_logging(level="INFO")  # reset


def test_setup_logging_adds_stream_handler():
    from bforbuntyai import setup_logging

    setup_logging(level="INFO")
    root = logging.getLogger("bforbuntyai")
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)


def test_setup_logging_writes_to_file(tmp_path):
    from bforbuntyai import setup_logging

    log_file = str(tmp_path / "bforbunty.log")
    setup_logging(level="INFO", file=log_file)
    logging.getLogger("bforbuntyai.test_file").info("file logging works")
    import os

    assert os.path.exists(log_file)
    assert "file logging works" in open(log_file).read()
    setup_logging(level="INFO")  # reset without file handler


# ─── Metrics ────────────────────────────────────────────────────────────────

def test_gan_metrics_after_training():
    pytest.importorskip("torch")
    from bforbuntyai import GAN

    ds = _fake_ds()
    gan = GAN(ds, latent_dim=16, batch_size=32)
    gan.train(epochs=2)
    assert hasattr(gan, "metrics")
    m = gan.metrics
    assert "g_loss" in m and "d_loss" in m
    assert len(m["g_loss"]) == 2
    assert len(m["d_loss"]) == 2


def test_dcgan_metrics_after_training():
    pytest.importorskip("torch")
    from bforbuntyai import DCGAN

    ds = _fake_ds()
    dcgan = DCGAN(ds, latent_dim=16, batch_size=32)
    dcgan.train(epochs=2)
    m = dcgan.metrics
    assert "g_loss" in m and "d_loss" in m
    assert len(m["g_loss"]) == 2


def test_vae_metrics_after_training():
    pytest.importorskip("torch")
    from bforbuntyai import VAE

    ds = _fake_ds()
    vae = VAE(ds, latent_dim=4)
    vae.train(epochs=2)
    m = vae.metrics
    assert "loss" in m
    assert len(m["loss"]) == 2


def test_conditional_gan_metrics_after_training():
    pytest.importorskip("torch")
    from bforbuntyai import ConditionalGAN

    ds = _fake_ds()
    cgan = ConditionalGAN(ds, num_classes=10, latent_dim=16, batch_size=32)
    cgan.train(epochs=2)
    m = cgan.metrics
    assert "g_loss" in m and "d_loss" in m


# ─── Universal HuggingFace Dataset ──────────────────────────────────────────

def test_huggingface_accepts_explicit_column():
    """HuggingFace dataset wrapper must honour an explicit text_column argument."""
    pytest.importorskip("datasets", exc_type=ImportError)
    from unittest.mock import MagicMock, patch

    mock_ds = MagicMock()
    mock_ds.column_names = ["title", "body", "score"]

    with patch("datasets.load_dataset", return_value=mock_ds):
        from bforbuntyai.datasets.huggingface import HuggingFace

        ds = HuggingFace("any/dataset", text_column="body")
        assert ds.text_column == "body"


def test_huggingface_fallback_to_first_column():
    """HuggingFace dataset falls back to the first column when no known column matches."""
    pytest.importorskip("datasets", exc_type=ImportError)
    from unittest.mock import MagicMock, patch

    mock_ds = MagicMock()
    mock_ds.column_names = ["headline", "category", "timestamp"]

    with patch("datasets.load_dataset", return_value=mock_ds):
        from bforbuntyai.datasets.huggingface import HuggingFace

        ds = HuggingFace("org/news-dataset")
        assert ds.text_column == "headline"
