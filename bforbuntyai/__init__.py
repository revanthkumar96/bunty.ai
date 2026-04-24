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
"""

from . import auth
from . import datasets as dataset
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

__version__ = "0.1.0"

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
]
