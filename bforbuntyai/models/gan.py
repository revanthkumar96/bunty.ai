from typing import Optional

import numpy as np

from .._base import BaseModel
from .._logging import get_logger
from .._utils import get_device, plot_grid

_logger = get_logger("models.gan")


class _Generator:
    pass  # forward ref


def _make_generator(latent_dim: int, img_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(128),
        nn.Linear(128, 256),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(256),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, img_dim),
        nn.Tanh(),
    )


def _make_discriminator(img_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(img_dim, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


class GAN(BaseModel):
    """Vanilla fully-connected GAN (PyTorch).

    Usage:
        from bforbuntyai import GAN, dataset
        gan = GAN(dataset.FashionMNIST())
        gan.train(epochs=50)
        gan.generate(n=25)
    """

    def __init__(self, dataset, latent_dim: int = 100, lr: float = 0.0002, batch_size: int = 128):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "GAN requires PyTorch.\nInstall with: pip install bforbuntyai[torch]"
            )

        self.dataset = dataset
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device()

        self.img_shape = dataset.shape          # (H, W, C)
        self.img_dim = int(np.prod(self.img_shape))

        self.G = _make_generator(latent_dim, self.img_dim).to(self.device)
        self.D = _make_discriminator(self.img_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.g_losses: list = []
        self.d_losses: list = []

    def train(self, epochs: int = 50, batch_size: Optional[int] = None) -> "GAN":
        import torch
        from tqdm import tqdm

        bs = batch_size or self.batch_size
        loader = self.dataset.as_torch_loader(split="train", batch_size=bs, gan=True)

        for epoch in range(epochs):
            g_sum = d_sum = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, _ in pbar:
                imgs = imgs.view(imgs.size(0), -1).to(self.device)
                n = imgs.size(0)
                valid = torch.ones(n, 1, device=self.device)
                fake = torch.zeros(n, 1, device=self.device)

                # Discriminator
                self.opt_D.zero_grad()
                z = torch.randn(n, self.latent_dim, device=self.device)
                gen = self.G(z)
                d_loss = (self.criterion(self.D(imgs), valid) + self.criterion(self.D(gen.detach()), fake)) / 2
                d_loss.backward()
                self.opt_D.step()

                # Generator
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
        imgs = imgs.reshape(-1, *self.img_shape)
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
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
            plt.title("GAN Training Losses")
            plt.show()

    @property
    def metrics(self) -> dict:
        return {"g_loss": self.g_losses, "d_loss": self.d_losses}

    def save(self, path: str = "gan.pth") -> None:
        import torch

        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)
        _logger.info("Saved to %s", path)

    def load(self, path: str = "gan.pth") -> "GAN":
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        return self
