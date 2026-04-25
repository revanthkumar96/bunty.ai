from typing import List, Optional, Union

import numpy as np

from .._base import BaseModel
from .._logging import get_logger
from ..auth import get_token

_logger = get_logger("models.image_captioner")


class ImageCaptioner(BaseModel):
    """Image captioning — generate text descriptions of images.

    Uses pipeline('image-to-text') so it works with any compatible model:
    BLIP, GIT, InstructBLIP, LLaVA, and others on HuggingFace Hub.

    Usage:
        from bforbuntyai import ImageCaptioner
        cap = ImageCaptioner()                                          # BLIP base (default)
        cap = ImageCaptioner("nlpconnect/vit-gpt2-image-captioning")   # ViT+GPT-2
        cap = ImageCaptioner("Salesforce/blip2-opt-2.7b")              # BLIP-2
        cap.caption("path/to/image.jpg")
        cap.caption("https://example.com/photo.jpg")
        cap.visualize(["img1.jpg", "img2.jpg"])
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "auto",
        token: Optional[str] = None,
    ):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "ImageCaptioner requires transformers.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        self.model_name = model_name
        self.token = get_token(token)
        self._device = self._resolve_device(device)

        load_kwargs: dict = {"model": model_name}
        if self.token:
            load_kwargs["token"] = self.token

        # Map device string to an integer device id for the pipeline API.
        # pipeline accepts device=-1 (CPU), device=0 (first CUDA), etc.
        if "cuda" in self._device:
            try:
                device_id = int(self._device.split(":")[-1]) if ":" in self._device else 0
            except ValueError:
                device_id = 0
            load_kwargs["device"] = device_id
        else:
            load_kwargs["device"] = -1

        _logger.info("Loading %s...", model_name)
        # transformers>=4.47 renamed "image-to-text" → "image-text-to-text"
        for task in ("image-text-to-text", "image-to-text"):
            try:
                self._pipe = pipeline(task, **load_kwargs)
                _logger.info("Ready (task=%s).", task)
                break
            except KeyError:
                continue
        else:
            raise RuntimeError(
                "Could not find a working image captioning pipeline task. "
                "Try: pip install -U transformers"
            )

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

    def _load_image(self, source: Union[str, np.ndarray]):
        from PIL import Image

        if isinstance(source, np.ndarray):
            return Image.fromarray((source * 255).astype(np.uint8)).convert("RGB")
        if isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        ):
            import io

            import requests

            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        return Image.open(source).convert("RGB")

    def caption(
        self,
        image: Union[str, np.ndarray],
        max_tokens: int = 50,
        num_beams: int = 5,
    ) -> str:
        img = self._load_image(image)
        result = self._pipe(img, max_new_tokens=max_tokens, num_beams=num_beams)
        text = result[0]["generated_text"]
        _logger.info("Caption: %s", text)
        return text

    def generate(self, images: Union[str, List], **kwargs) -> Union[str, List[str]]:
        if isinstance(images, (str, np.ndarray)):
            return self.caption(images, **kwargs)
        return [self.caption(img, **kwargs) for img in images]

    def visualize(self, images: List[Union[str, np.ndarray]], **kwargs) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
        if len(images) == 1:
            axes = [axes]
        for ax, src in zip(axes, images):
            img = self._load_image(src)
            cap = self.caption(src, **kwargs)
            ax.imshow(img)
            ax.set_title(cap, fontsize=9, wrap=True)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def train(self, **kwargs) -> "ImageCaptioner":
        raise NotImplementedError("ImageCaptioner is inference-only in bforbuntyai.")
