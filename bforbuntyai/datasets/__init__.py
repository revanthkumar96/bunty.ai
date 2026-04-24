from .cifar import CIFAR10
from .custom import Custom
from .huggingface import HuggingFace
from .mnist import FashionMNIST, MNIST
from .pix2pix import Edges2Shoes

__all__ = ["MNIST", "FashionMNIST", "CIFAR10", "HuggingFace", "Edges2Shoes", "Custom"]
