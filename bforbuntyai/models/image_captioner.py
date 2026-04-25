from typing import List, Optional, Union

import numpy as np

from .._base import BaseModel
from .._logging import get_logger
from ..auth import get_token

_logger = get_logger("models.image_captioner")


class ImageCaptioner(BaseModel):
    """Image captioning using BLIP directly (no pipeline).

    Uses BlipProcessor + BlipForConditionalGeneration for reliable inference
    across all transformers versions, with optional conditional captioning.

    Usage:
        from bforbuntyai import ImageCaptioner

        cap = ImageCaptioner()                                        # BLIP-base (default)
        cap = ImageCaptioner("Salesforce/blip-image-captioning-large") # BLIP-large
        cap = ImageCaptioner("Salesforce/blip2-opt-2.7b")             # BLIP-2 (needs GPU)

        cap.caption("photo.jpg")
        cap.caption("https://example.com/photo.jpg")
        cap.caption("photo.jpg", prompt="a photo of")   # conditional captioning
        cap.visualize(["img1.jpg", "img2.jpg", "img3.jpg"])
    """

    DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        token: Optional[str] = None,
    ):
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ImportError(
                "ImageCaptioner requires transformers.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        self.model_name = model_name
        self.token = get_token(token)
        self._device = self._resolve_device(device)

        load_kwargs: dict = {}
        if self.token:
            load_kwargs["token"] = self.token

        _logger.info("Loading %s...", model_name)
        self._processor = BlipProcessor.from_pretrained(model_name, **load_kwargs)
        self._model = BlipForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        ).to(self._device)
        self._model.eval()
        _logger.info("Ready on %s.", self._device)

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
        prompt: Optional[str] = None,
        max_tokens: int = 50,
        num_beams: int = 5,
        min_tokens: int = 5,
    ) -> str:
        """Generate a caption for a single image.

        Args:
            image:      Local path, URL, or numpy array (H, W, C) in [0, 1].
            prompt:     Optional text prompt for conditional captioning
                        e.g. "a photo of" → "a photo of a cat on a table".
            max_tokens: Maximum number of new tokens to generate.
            num_beams:  Beam search width; higher = better quality but slower.
            min_tokens: Minimum new tokens — prevents empty outputs.
        """
        import torch

        img = self._load_image(image)

        if prompt:
            inputs = self._processor(img, prompt, return_tensors="pt").to(self._device)
        else:
            inputs = self._processor(img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_beams=num_beams,
            )

        text = self._processor.decode(output_ids[0], skip_special_tokens=True)
        _logger.info("Caption: %s", text)
        return text

    def generate(self, images: Union[str, List], **kwargs) -> Union[str, List[str]]:
        """Caption one image or a list of images."""
        if isinstance(images, (str, np.ndarray)):
            return self.caption(images, **kwargs)
        return [self.caption(img, **kwargs) for img in images]

    def visualize(
        self,
        images: List[Union[str, np.ndarray]],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Display images in a grid with their captions as titles."""
        import matplotlib.pyplot as plt

        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
        if n == 1:
            axes = [axes]
        for ax, src in zip(axes, images):
            img = self._load_image(src)
            cap = self.caption(src, prompt=prompt, **kwargs)
            ax.imshow(img)
            ax.set_title(cap, fontsize=9, wrap=True)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def train(self, **kwargs) -> "ImageCaptioner":
        raise NotImplementedError("ImageCaptioner is inference-only in bforbuntyai.")
