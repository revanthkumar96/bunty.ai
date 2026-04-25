"""bforbuntyai — GenAI experiments in a few lines of code.

Quick start:
    from bforbuntyai import GAN, dataset
    gan = GAN(dataset.FashionMNIST())
    gan.train(epochs=50)
    gan.generate(n=25)

HuggingFace auth:
    from bforbuntyai import auth
    auth.login()                 # interactive
    auth.login(token="hf_...")   # explicit token

Logging:
    from bforbuntyai import setup_logging
    setup_logging(level="WARNING")   # silence training output
    setup_logging(level="DEBUG")     # verbose
    setup_logging(file="run.log")    # also write to a file
"""

from . import auth
from . import datasets as dataset
from ._logging import setup_logging
from .models import (
    AutoEncoder,
    ConditionalGAN,
    DCGAN,
    EthicalEvaluator,
    GAN,
    ImageCaptioner,
    Pix2Pix,
    StableDiffusion,
    TextFineTuner,
    TextGenerator,
    VAE,
)

__version__ = "0.2.0"

__all__ = [
    # Models
    "AutoEncoder",
    "GAN",
    "DCGAN",
    "ConditionalGAN",
    "VAE",
    "Pix2Pix",
    "TextGenerator",
    "TextFineTuner",
    "StableDiffusion",
    "ImageCaptioner",
    "EthicalEvaluator",
    # Namespaces
    "dataset",
    "auth",
    # Logging
    "setup_logging",
]

# Enable INFO logging by default so training progress is visible out of the box.
setup_logging(level="INFO")
