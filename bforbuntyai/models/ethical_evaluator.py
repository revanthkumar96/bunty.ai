from typing import Dict, List, Optional, Union

from .._base import BaseModel
from ..auth import get_token

_TOXICITY_THRESHOLD = 0.5
_BIAS_THRESHOLD = 0.6


class EthicalEvaluator(BaseModel):
    """Evaluate text for toxicity and bias.

    Combines Detoxify (6-category toxicity scoring) with a HuggingFace bias classifier.

    Usage:
        from bforbuntyai import EthicalEvaluator
        ev = EthicalEvaluator()
        ev.evaluate("Some text to check")
        ev.evaluate_batch(["text one", "text two"])
    """

    def __init__(
        self,
        toxicity_model: str = "original",
        bias_model: str = "himel7/bias-detector",
        token: Optional[str] = None,
    ):
        self.token = get_token(token)
        self._toxicity_model_name = toxicity_model
        self._bias_model_name = bias_model
        self._tox = None
        self._bias = None

    def _load_toxicity(self):
        if self._tox is None:
            try:
                from detoxify import Detoxify

                self._tox = Detoxify(self._toxicity_model_name)
            except ImportError:
                raise ImportError(
                    "Toxicity detection requires detoxify.\n"
                    "Install with: pip install bforbuntyai[ethics]"
                )
        return self._tox

    def _load_bias(self):
        if self._bias is None:
            try:
                from transformers import pipeline

                load_kwargs: dict = {}
                if self.token:
                    load_kwargs["token"] = self.token
                self._bias = pipeline(
                    "text-classification",
                    model=self._bias_model_name,
                    **load_kwargs,
                )
            except ImportError:
                raise ImportError(
                    "Bias detection requires transformers.\n"
                    "Install with: pip install bforbuntyai[ethics]"
                )
        return self._bias

    def evaluate(self, text: str) -> Dict:
        tox_scores = self._load_toxicity().predict(text)
        bias_out = self._load_bias()(text, truncation=True)[0]

        is_toxic = any(v >= _TOXICITY_THRESHOLD for v in tox_scores.values())
        is_biased = bias_out["label"].lower() == "biased" and bias_out["score"] >= _BIAS_THRESHOLD
        verdict = "POTENTIALLY HARMFUL" if (is_toxic or is_biased) else "SAFE"

        result = {
            "text": text,
            "verdict": verdict,
            "toxicity": {k: round(float(v), 4) for k, v in tox_scores.items()},
            "bias": {"label": bias_out["label"], "score": round(float(bias_out["score"]), 4)},
        }
        self._print_report(result)
        return result

    @staticmethod
    def _print_report(r: Dict) -> None:
        print(f"\n{'=' * 50}")
        print(f"Text: {r['text'][:80]}{'...' if len(r['text']) > 80 else ''}")
        print(f"Verdict: {r['verdict']}")
        print("\nToxicity Scores:")
        for cat, score in r["toxicity"].items():
            flag = " ⚠" if score >= _TOXICITY_THRESHOLD else ""
            print(f"  {cat:<25} {score:.4f}{flag}")
        print(f"\nBias: {r['bias']['label']} (confidence: {r['bias']['score']:.4f})")
        print("=" * 50)

    def evaluate_batch(self, texts: List[str]) -> List[Dict]:
        return [self.evaluate(t) for t in texts]

    def generate(self, texts: Union[str, List[str]], **kwargs) -> Union[Dict, List[Dict]]:
        if isinstance(texts, str):
            return self.evaluate(texts)
        return self.evaluate_batch(texts)

    def visualize(self, texts: List[str]) -> None:
        self.evaluate_batch(texts)

    def train(self, **kwargs) -> "EthicalEvaluator":
        raise NotImplementedError("EthicalEvaluator is inference-only.")
