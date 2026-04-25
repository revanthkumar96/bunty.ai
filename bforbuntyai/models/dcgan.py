from typing import Optional

import numpy as np

from .._base import BaseModel
from .._logging import get_logger
from .._utils import get_device, plot_grid

_logger = get_logger("models.dcgan")


def _make_dc_generator(latent_dim: int, channels: int, img_size: int):
    import torch.nn as nn

    start = img_size // 4  # 7 for 28px, 8 for 32px
    return nn.Sequential(
        nn.Linear(latent_dim, 128 * start * start),
        _Reshape(128, start, start),
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(64, channels, 3, padding=1),
        nn.Tanh(),
    )


def _make_dc_discriminator(channels: int, img_size: int):
    import torch.nn as nn

    ds = img_size // 4  # size after 2x stride-2 convolutions

    def _block(in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers += [nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        return layers

    return nn.Sequential(
        *_block(channels, 32, normalize=False),
        *_block(32, 64),
        nn.Flatten(),
        nn.Linear(64 * ds * ds, 1),
        nn.Sigmoid(),
    )


class _Reshape(object):
    pass  # replaced below


import torch.nn as nn  # noqa: E402


class _Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class DCGAN(BaseModel):
    """Deep Convolutional GAN (PyTorch).

    Optimised for 28×28 and 32×32 images (MNIST, FashionMNIST, CIFAR-10).

    Usage:
        from bforbuntyai import DCGAN, dataset
        gan = DCGAN(dataset.MNIST())
        gan.train(epochs=20)
        gan.generate(n=25)
    """

    def __init__(self, dataset, latent_dim: int = 100, lr: float = 0.0002, batch_size: int = 64):
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device()

        channels = dataset.shape[-1]
        img_size = dataset.shape[0]

        self.G = _make_dc_generator(latent_dim, channels, img_size).to(self.device)
        self.D = _make_dc_discriminator(channels, img_size).to(self.device)

        import torch.optim as optim

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

        import torch.nn as nn

        self.criterion = nn.BCELoss()
        self.img_shape = dataset.shape
        self.g_losses: list = []
        self.d_losses: list = []

    def train(self, epochs: int = 20, batch_size: Optional[int] = None) -> "DCGAN":
        import torch
        from tqdm import tqdm

        bs = batch_size or self.batch_size
        loader = self.dataset.as_torch_loader(split="train", batch_size=bs, gan=True)

        for epoch in range(epochs):
            g_sum = d_sum = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, _ in pbar:
                imgs = imgs.to(self.device)
                n = imgs.size(0)
                valid = torch.ones(n, 1, device=self.device)
                fake = torch.zeros(n, 1, device=self.device)

                self.opt_D.zero_grad()
                z = torch.randn(n, self.latent_dim, device=self.device)
                gen = self.G(z)
                d_loss = (self.criterion(self.D(imgs), valid) + self.criterion(self.D(gen.detach()), fake)) / 2
                d_loss.backward()
                self.opt_D.step()

                self.opt_G.zero_grad()
                z = torch.randn(n, self.latent_dim, device=self.device)
                g_loss = self.criterion(self.D(self.G(z)), valid)
                g_loss.backward()
                self.opt_G.step()

                g_sum += g_loss.item()
                d_sum += d_loss.item()
                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            self.g_losses.append(g_sum / len(loader))
            self.d_losses.append(d_sum / len(loader))
            _logger.info(
                "Epoch %d/%d  G: %.4f  D: %.4f",
                epoch + 1, epochs, self.g_losses[-1], self.d_losses[-1],
            )

        return self

    def generate(self, n: int = 25, return_array: bool = False) -> np.ndarray:
        import torch

        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            imgs = self.G(z).cpu().numpy()
        imgs = np.transpose(imgs, (0, 2, 3, 1))  # (N,C,H,W) → (N,H,W,C)
        imgs = (imgs + 1) / 2
        if not return_array:
            plot_grid(imgs, cols=min(n, 5))
        return imgs

    def visualize(self) -> None:
        import matplotlib.pyplot as plt

        self.generate(n=25)
        if self.g_losses:
            plt.figure(figsize=(10, 4))
            plt.plot(self.g_losses, label="Generator")
            plt.plot(self.d_losses, label="Discriminator")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("DCGAN Training Losses")
            plt.show()

    @property
    def metrics(self) -> dict:
        return {"g_loss": self.g_losses, "d_loss": self.d_losses}

    def save(self, path: str = "dcgan.pth") -> None:
        import torch

        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)
        _logger.info("Saved to %s", path)

    def load(self, path: str = "dcgan.pth") -> "DCGAN":
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        return self
