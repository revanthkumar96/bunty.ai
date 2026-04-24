from typing import Optional

import numpy as np

from .._base import BaseModel
from .._utils import get_device, plot_grid


class Pix2Pix(BaseModel):
    """Pix2Pix image-to-image translation with U-Net generator + PatchGAN discriminator (PyTorch).

    Works with the Edges2Shoes dataset out of the box.
    Also accepts any dataset with an .as_torch_loader() that yields (edge, real) pairs.

    Usage:
        from bforbuntyai import Pix2Pix, dataset
        p2p = Pix2Pix(dataset.Edges2Shoes())
        p2p.train(epochs=5)
        p2p.visualize()
    """

    def __init__(
        self,
        dataset,
        lr: float = 0.0002,
        batch_size: int = 16,
        lambda_l1: float = 100.0,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("Pix2Pix requires PyTorch. pip install bforbuntyai[torch]")

        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_l1 = lambda_l1
        self.device = get_device()

        self.G = _UNetGenerator().to(self.device)
        self.D = _PatchDiscriminator().to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

        import torch.nn as nn

        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()

        self.g_losses: list = []
        self.d_losses: list = []

    def train(self, epochs: int = 5, batch_size: Optional[int] = None) -> "Pix2Pix":
        import torch
        from tqdm import tqdm

        bs = batch_size or self.batch_size
        loader = self.dataset.as_torch_loader(split="train", batch_size=bs, gan=True)

        for epoch in range(epochs):
            g_sum = d_sum = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for real_A, real_B in pbar:
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                n = real_A.size(0)
                ones = torch.ones(n, 1, 30, 30, device=self.device)
                zeros = torch.zeros(n, 1, 30, 30, device=self.device)

                fake_B = self.G(real_A)

                # Discriminator
                self.opt_D.zero_grad()
                d_real = self.criterion_gan(self.D(real_A, real_B), ones)
                d_fake = self.criterion_gan(self.D(real_A, fake_B.detach()), zeros)
                d_loss = (d_real + d_fake) * 0.5
                d_loss.backward()
                self.opt_D.step()

                # Generator
                self.opt_G.zero_grad()
                g_gan = self.criterion_gan(self.D(real_A, fake_B), ones)
                g_l1 = self.criterion_l1(fake_B, real_B) * self.lambda_l1
                g_loss = g_gan + g_l1
                g_loss.backward()
                self.opt_G.step()

                g_sum += g_loss.item()
                d_sum += d_loss.item()
                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            self.g_losses.append(g_sum / len(loader))
            self.d_losses.append(d_sum / len(loader))
            print(f"Epoch {epoch + 1}/{epochs}  G: {self.g_losses[-1]:.4f}  D: {self.d_losses[-1]:.4f}")

        return self

    def translate(self, edge: np.ndarray) -> np.ndarray:
        import torch

        self.G.eval()
        if edge.ndim == 3:
            edge = edge[np.newaxis]
        x = torch.FloatTensor(np.transpose(edge, (0, 3, 1, 2))).to(self.device) * 2 - 1
        with torch.no_grad():
            out = self.G(x).cpu().numpy()
        out = np.transpose(out, (0, 2, 3, 1))
        return (out + 1) / 2

    def generate(self, n: int = 5, split: str = "test") -> np.ndarray:
        import torch

        loader = self.dataset.as_torch_loader(split=split, batch_size=n, gan=True)
        real_A, real_B = next(iter(loader))
        real_A = real_A[:n].to(self.device)
        real_B = real_B[:n]

        self.G.eval()
        with torch.no_grad():
            fake_B = self.G(real_A).cpu().numpy()

        A_np = (np.transpose(real_A.cpu().numpy(), (0, 2, 3, 1)) + 1) / 2
        B_np = (real_B.numpy().transpose(0, 2, 3, 1) + 1) / 2
        fake_np = (np.transpose(fake_B, (0, 2, 3, 1)) + 1) / 2

        images = []
        titles = []
        for i in range(n):
            images += [A_np[i], fake_np[i], B_np[i]]
            titles += ["Input (Edge)", "Generated", "Real Photo"]
        plot_grid(images, titles=titles, cols=3)
        return fake_np

    def visualize(self) -> None:
        self.generate(n=5)

    def save(self, path: str = "pix2pix.pth") -> None:
        import torch

        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()}, path)
        print(f"Saved to {path}")

    def load(self, path: str = "pix2pix.pth") -> "Pix2Pix":
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        return self


# ── U-Net Generator ────────────────────────────────────────────────────────────

import torch.nn as nn  # noqa: E402


class _EncBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _UNetGenerator(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 3, ngf: int = 64):
        super().__init__()
        self.e1 = _EncBlock(in_c, ngf, normalize=False)
        self.e2 = _EncBlock(ngf, ngf * 2)
        self.e3 = _EncBlock(ngf * 2, ngf * 4)
        self.e4 = _EncBlock(ngf * 4, ngf * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.ReLU(inplace=True)
        )
        self.d1 = _DecBlock(ngf * 8, ngf * 8, dropout=True)
        self.d2 = _DecBlock(ngf * 16, ngf * 4)
        self.d3 = _DecBlock(ngf * 8, ngf * 2)
        self.d4 = _DecBlock(ngf * 4, ngf)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_c, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        import torch

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bottleneck(e4)
        d1 = torch.cat([self.d1(b), e4], 1)
        d2 = torch.cat([self.d2(d1), e3], 1)
        d3 = torch.cat([self.d3(d2), e2], 1)
        d4 = torch.cat([self.d4(d3), e1], 1)
        return self.out(d4)


class _PatchDiscriminator(nn.Module):
    def __init__(self, in_c: int = 6, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        import torch

        return self.model(torch.cat([img_A, img_B], 1))
