from typing import Optional, Tuple

import numpy as np

from .._base import BaseModel
from .._utils import get_device, plot_grid


class VAE(BaseModel):
    """Variational AutoEncoder (PyTorch).

    Use latent_dim=2 to get a 2-D latent space you can visualise with .visualize_latent().
    Use latent_dim=20 for higher-quality reconstruction and interpolation.

    Usage:
        from bforbuntyai import VAE, dataset
        vae = VAE(dataset.MNIST(), latent_dim=2)
        vae.train(epochs=20)
        vae.visualize()                  # reconstructions
        vae.visualize_latent()           # 2-D scatter (only when latent_dim=2)
        vae.interpolate(img_a, img_b)    # morph between two images
    """

    def __init__(
        self,
        dataset,
        latent_dim: int = 2,
        hidden_dim: int = 512,
        lr: float = 1e-3,
        batch_size: int = 128,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("VAE requires PyTorch. pip install bforbuntyai[torch]")

        self.dataset = dataset
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device()

        self.img_shape = dataset.shape
        self.img_dim = int(np.prod(self.img_shape))

        self.encoder, self.fc_mu, self.fc_logvar, self.decoder = self._build(nn)
        params = (
            list(self.encoder.parameters())
            + list(self.fc_mu.parameters())
            + list(self.fc_logvar.parameters())
            + list(self.decoder.parameters())
        )
        self.optimizer = optim.Adam(params, lr=lr)
        self.losses: list = []

    def _build(self, nn):
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 400),
            nn.ReLU(),
        ).to(self.device)
        fc_mu = nn.Linear(400, self.latent_dim).to(self.device)
        fc_logvar = nn.Linear(400, self.latent_dim).to(self.device)
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.img_dim),
            nn.Sigmoid(),
        ).to(self.device)
        return encoder, fc_mu, fc_logvar, decoder

    def _encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(self, mu, logvar):
        import torch

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z):
        return self.decoder(z)

    def _loss(self, recon, x, mu, logvar):
        import torch.nn.functional as F

        bce = F.binary_cross_entropy(recon, x.view(-1, self.img_dim), reduction="sum")
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return bce + kld

    def train(self, epochs: int = 20, batch_size: Optional[int] = None) -> "VAE":
        from tqdm import tqdm

        bs = batch_size or self.batch_size
        loader = self.dataset.as_torch_loader(split="train", batch_size=bs, gan=False)

        for epoch in range(epochs):
            total = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, _ in pbar:
                imgs = imgs.to(self.device)
                mu, logvar = self._encode(imgs)
                z = self._reparameterize(mu, logvar)
                recon = self._decode(z)
                loss = self._loss(recon, imgs, mu, logvar)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item() / len(imgs):.2f}")

            avg = total / len(loader.dataset)
            self.losses.append(avg)
            print(f"Epoch {epoch + 1}/{epochs}  loss: {avg:.4f}")

        return self

    def generate(self, n: int = 16, return_array: bool = False) -> np.ndarray:
        import torch

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self.device)
            imgs = self._decode(z).cpu().numpy()
        imgs = imgs.reshape(-1, *self.img_shape)
        if not return_array:
            plot_grid(imgs, cols=min(n, 8))
        return imgs

    def visualize(self, n: int = 10) -> None:
        import torch

        x_test, _ = self.dataset.as_numpy("test")
        import torch

        x_t = torch.FloatTensor(np.transpose(x_test[:n], (0, 3, 1, 2))).to(self.device)
        with torch.no_grad():
            mu, logvar = self._encode(x_t)
            z = self._reparameterize(mu, logvar)
            recon = self._decode(z).cpu().numpy().reshape(-1, *self.img_shape)

        orig = [x_test[i] for i in range(n)]
        rec = [recon[i] for i in range(n)]
        plot_grid(orig + rec, titles=[f"Orig {i}" for i in range(n)] + [f"Recon {i}" for i in range(n)], cols=n)

    def visualize_latent(self, n: int = 1000) -> None:
        if self.latent_dim != 2:
            print(f"visualize_latent() only works when latent_dim=2 (current: {self.latent_dim})")
            return
        import matplotlib.pyplot as plt
        import torch

        x, y = self.dataset.as_numpy("test")
        x = x[:n]
        y = y[:n]
        x_t = torch.FloatTensor(np.transpose(x, (0, 3, 1, 2))).to(self.device)
        with torch.no_grad():
            mu, _ = self._encode(x_t)
        mu_np = mu.cpu().numpy()

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(mu_np[:, 0], mu_np[:, 1], c=y, cmap="tab10", alpha=0.7, s=10)
        plt.colorbar(scatter, label="Class")
        plt.title("VAE Latent Space")
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

    def interpolate(self, img_a: np.ndarray, img_b: np.ndarray, steps: int = 10) -> np.ndarray:
        import torch

        def _prep(img):
            if img.ndim == 2:
                img = img[..., np.newaxis]
            t = torch.FloatTensor(np.transpose(img[np.newaxis], (0, 3, 1, 2))).to(self.device)
            return t

        with torch.no_grad():
            mu_a, _ = self._encode(_prep(img_a))
            mu_b, _ = self._encode(_prep(img_b))
            interps = [
                self._decode(mu_a + (mu_b - mu_a) * t).cpu().numpy().reshape(*self.img_shape)
                for t in np.linspace(0, 1, steps)
            ]
        plot_grid(interps, cols=steps, figsize=(steps * 1.5, 2))
        return np.stack(interps)

    def save(self, path: str = "vae.pth") -> None:
        import torch

        torch.save({
            "encoder": self.encoder.state_dict(),
            "fc_mu": self.fc_mu.state_dict(),
            "fc_logvar": self.fc_logvar.state_dict(),
            "decoder": self.decoder.state_dict(),
        }, path)
        print(f"Saved to {path}")

    def load(self, path: str = "vae.pth") -> "VAE":
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.fc_mu.load_state_dict(ckpt["fc_mu"])
        self.fc_logvar.load_state_dict(ckpt["fc_logvar"])
        self.decoder.load_state_dict(ckpt["decoder"])
        return self
