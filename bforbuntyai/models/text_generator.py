from typing import List, Optional

from .._base import BaseModel
from .._logging import get_logger
from ..auth import get_token

_logger = get_logger("models.text_generator")


class TextGenerator(BaseModel):
    """Inference-only text generator — works with any AutoModelForCausalLM-compatible model.

    Usage:
        from bforbuntyai import TextGenerator
        gen = TextGenerator()                              # GPT-2 (default)
        gen = TextGenerator("EleutherAI/gpt-neo-125M")   # GPT-Neo
        gen = TextGenerator("mistralai/Mistral-7B-v0.1") # Mistral (needs token)
        gen.generate("Once upon a time")
        gen.generate("The future of AI is", max_tokens=200, num_return=3)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        token: Optional[str] = None,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "TextGenerator requires transformers.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        self.model_name = model_name
        self.token = get_token(token)

        load_kwargs: dict = {}
        if self.token:
            load_kwargs["token"] = self.token

        _logger.info("Loading %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._device = self._resolve_device(device)
        self._model = self._model.to(self._device)
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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return: int = 3,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        for i, t in enumerate(texts, 1):
            _logger.info("--- Sample %d ---\n%s", i, t)
        return texts

    def visualize(self, prompt: str = "Once upon a time", **kwargs) -> None:
        self.generate(prompt, **kwargs)

    def train(self, **kwargs) -> "TextGenerator":
        raise NotImplementedError("Use TextFineTuner for training. TextGenerator is inference-only.")
