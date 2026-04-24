from typing import List, Optional

import numpy as np

from .._base import BaseModel
from .._utils import get_device, plot_grid


class ConditionalGAN(BaseModel):
    """Conditional GAN — generate images conditioned on class label (PyTorch).

    Usage:
        from bforbuntyai import ConditionalGAN, dataset
        cgan = ConditionalGAN(dataset.MNIST(), num_classes=10)
        cgan.train(epochs=50)
        cgan.generate_class(labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """

    def __init__(
        self,
        dataset,
        num_classes: int = 10,
        latent_dim: int = 100,
        embed_dim: int = 10,
        lr: float = 0.0002,
        batch_size: int = 128,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("ConditionalGAN requires PyTorch. pip install bforbuntyai[torch]")

        self.dataset = dataset
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device()

        self.img_shape = dataset.shape
        self.img_dim = int(np.prod(self.img_shape))

        self.G = self._build_generator(nn, latent_dim, embed_dim, num_classes, self.img_dim).to(self.device)
        self.D = self._build_discriminator(nn, embed_dim, num_classes, self.img_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.g_losses: list = []
        self.d_losses: list = []

    @staticmethod
    def _build_generator(nn, latent_dim, embed_dim, num_classes, img_dim):
        class _G(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_emb = nn.Embedding(num_classes, embed_dim)
                self.net = nn.Sequential(
                    nn.Linear(latent_dim + embed_dim, 256),
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

            def forward(self, z, labels):
                import torch

                x = torch.cat([z, self.label_emb(labels)], dim=1)
                return self.net(x)

        return _G()

    @staticmethod
    def _build_discriminator(nn, embed_dim, num_classes, img_dim):
        class _D(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_emb = nn.Embedding(num_classes, embed_dim)
                self.net = nn.Sequential(
                    nn.Linear(img_dim + embed_dim, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def forward(self, imgs, labels):
                import torch

                x = torch.cat([imgs, self.label_emb(labels)], dim=1)
                return self.net(x)

        return _D()

    def train(self, epochs: int = 50, batch_size: Optional[int] = None) -> "ConditionalGAN":
        import torch
        from tqdm import tqdm

        bs = batch_size or self.batch_size
        loader = self.dataset.as_torch_loader(split="train", batch_size=bs, gan=True)

        for epoch in range(epochs):
            g_sum = d_sum = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, labels in pbar:
                imgs = imgs.view(imgs.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                n = imgs.size(0)
                valid = torch.ones(n, 1, device=self.device)
                fake_t = torch.zeros(n, 1, device=self.device)

                self.opt_D.zero_grad()
                z = torch.randn(n, self.latent_dim, device=self.device)
                gen = self.G(z, labels)
                d_loss = (
                    self.criterion(self.D(imgs, labels), valid)
                    + self.criterion(self.D(gen.detach(), labels), fake_t)
                ) / 2
                d_loss.backward()
                self.opt_D.step()

                self.opt_G.zero_grad()
                z = torch.randn(n, self.latent_dim, device=self.device)
                gen = self.G(z, labels)
                g_loss = self.criterion(self.D(gen, labels), valid)
                g_loss.backward()
                self.opt_G.step()

                g_sum += g_loss.item()
                d_sum += d_loss.item()
                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            self.g_losses.append(g_sum / len(loader))
            self.d_losses.append(d_sum / len(loader))
            print(f"Epoch {epoch + 1}/{epochs}  G: {self.g_losses[-1]:.4f}  D: {self.d_losses[-1]:.4f}")

        return self

    def generate_class(self, labels: Optional[List[int]] = None) -> np.ndarray:
        import torch

        if labels is None:
            labels = list(range(self.num_classes))
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(len(labels), self.latent_dim, device=self.device)
            lbl_t = torch.LongTensor(labels).to(self.device)
            imgs = self.G(z, lbl_t).cpu().numpy()
        imgs = imgs.reshape(-1, *self.img_shape)
        imgs = (imgs + 1) / 2
        cls_names = getattr(self.dataset, "class_names", [str(l) for l in labels])
        titles = [cls_names[l] if l < len(cls_names) else str(l) for l in labels]
        plot_grid(imgs, titles=titles, cols=len(labels))
        return imgs

    def generate(self, n: int = 10, **kwargs) -> np.ndarray:
        return self.generate_class(labels=list(range(min(n, self.num_classes))))

    def visualize(self) -> None:
        self.generate_class()

    def save(self, path: str = "cgan.pth") -> None:
        import torch

        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)
        print(f"Saved to {path}")

    def load(self, path: str = "cgan.pth") -> "ConditionalGAN":
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        return self
