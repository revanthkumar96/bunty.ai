from typing import Optional

from ..auth import get_token


class HuggingFace:
    """Thin wrapper around a HuggingFace dataset.

    Usage:
        data = HuggingFace("amazon_polarity", split="train[:200]")
        data = HuggingFace("imdb", split="train")
        data = HuggingFace("private/dataset", token="hf_...")
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: Optional[str] = None,
        token: Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets is required.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        resolved_token = get_token(token)
        self.dataset_name = dataset_name
        self.split = split
        self.token = resolved_token

        load_kwargs: dict = {"split": split}
        if resolved_token:
            load_kwargs["token"] = resolved_token

        self.hf_dataset = load_dataset(dataset_name, **load_kwargs)
        self.text_column = text_column or self._detect_text_column()
        self.name = dataset_name

    def _detect_text_column(self) -> str:
        candidates = ["text", "content", "sentence", "review", "document", "body"]
        cols = self.hf_dataset.column_names if hasattr(self.hf_dataset, "column_names") else []
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if cols else "text"

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        return f"HuggingFace('{self.dataset_name}', split='{self.split}', n={len(self)})"
