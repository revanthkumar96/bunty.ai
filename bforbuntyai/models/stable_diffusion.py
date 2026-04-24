from typing import List, Optional

import numpy as np

from .._base import BaseModel
from .._utils import plot_grid
from ..auth import get_token


class StableDiffusion(BaseModel):
    """Stable Diffusion text-to-image generation (inference only).

    Requires a HuggingFace token for gated model versions.

    Usage:
        from bforbuntyai import StableDiffusion, auth
        auth.login()                          # if using a gated model
        sd = StableDiffusion()
        sd.generate("A sunset over mountains", n=4)
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        token: Optional[str] = None,
    ):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError(
                "StableDiffusion requires diffusers.\n"
                "Install with: pip install bforbuntyai[diffusers]"
            )

        self.model_id = model_id
        self.token = get_token(token)
        self._device = self._resolve_device(device)

        print(f"Loading {model_id} on {self._device}...")
        dtype = torch.float16 if "cuda" in self._device else torch.float32

        load_kwargs: dict = {"torch_dtype": dtype}
        if self.token:
            load_kwargs["token"] = self.token

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        self.pipe = self.pipe.to(self._device)

        if "cuda" in self._device:
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        print("Ready.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 4,
        steps: int = 20,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = 42,
        save_dir: Optional[str] = None,
    ) -> List:
        import torch

        if negative_prompt is None:
            negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, watermark, text"

        generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None
        results = self.pipe(
            prompt=[prompt] * n,
            negative_prompt=[negative_prompt] * n,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        images = results.images

        if save_dir:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                p = Path(save_dir) / f"generated_{i}.png"
                img.save(str(p))
                print(f"Saved {p}")

        imgs_np = [np.array(img) / 255.0 for img in images]
        plot_grid(imgs_np, cols=n)
        return images

    def visualize(self, prompt: str = "A beautiful landscape, 4K photo", **kwargs) -> None:
        self.generate(prompt, **kwargs)

    def train(self, **kwargs) -> "StableDiffusion":
        raise NotImplementedError("StableDiffusion is inference-only in bforbuntyai.")
