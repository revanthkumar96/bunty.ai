from typing import List, Optional

from ..auth import get_token
from .._logging import get_logger

_logger = get_logger("datasets.huggingface")

# Common text-column names, tried in priority order.
_TEXT_COLUMN_CANDIDATES: List[str] = [
    "text", "content", "sentence", "review", "document", "body",
    "abstract", "description", "title", "article", "passage",
    "question", "answer", "context", "summary", "caption",
    "transcript", "dialogue", "comment", "message", "post",
]


class HuggingFace:
    """Thin wrapper around any HuggingFace dataset.

    Works with any dataset on HuggingFace Hub — public or private, text or
    structured. Pass text_column= to pin a specific column; otherwise the
    wrapper auto-detects it.

    Usage:
        data = HuggingFace("imdb")
        data = HuggingFace("amazon_polarity", split="train[:200]")
        data = HuggingFace("my-org/my-dataset", token="hf_...")
        data = HuggingFace("my-org/news", text_column="headline")
        data = HuggingFace("squad", text_column="context")
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

        _logger.info("Loading dataset '%s' (split=%s)...", dataset_name, split)
        self.hf_dataset = load_dataset(dataset_name, **load_kwargs)
        self.text_column = text_column or self._detect_text_column()
        self.name = dataset_name
        _logger.info(
            "Loaded %d examples. Text column: '%s'", len(self), self.text_column
        )

    def _detect_text_column(self) -> str:
        cols: List[str] = (
            list(self.hf_dataset.column_names)
            if hasattr(self.hf_dataset, "column_names")
            else []
        )
        for candidate in _TEXT_COLUMN_CANDIDATES:
            if candidate in cols:
                return candidate
        if cols:
            _logger.debug(
                "No known text column found in %s; defaulting to first column '%s'",
                cols, cols[0],
            )
            return cols[0]
        return "text"

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        return f"HuggingFace('{self.dataset_name}', split='{self.split}', n={len(self)})"
