from typing import List, Optional, Union

import numpy as np

from .._base import BaseModel
from ..auth import get_token


class ImageCaptioner(BaseModel):
    """BLIP image captioning — generate text descriptions of images.

    Usage:
        from bforbuntyai import ImageCaptioner
        cap = ImageCaptioner()
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
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ImportError(
                "ImageCaptioner requires transformers.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        self.token = get_token(token)
        self._device = self._resolve_device(device)

        load_kwargs: dict = {}
        if self.token:
            load_kwargs["token"] = self.token

        print(f"Loading {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name, **load_kwargs)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        self._model = self._model.to(self._device)
        self._model.eval()
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

    def _load_image(self, source: Union[str, np.ndarray]):
        from PIL import Image

        if isinstance(source, np.ndarray):
            return Image.fromarray((source * 255).astype(np.uint8)).convert("RGB")
        if source.startswith("http://") or source.startswith("https://"):
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
        import torch

        img = self._load_image(image)
        inputs = self.processor(img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, num_beams=num_beams, early_stopping=True
            )
        text = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"Caption: {text}")
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
