"""Text model and ethical evaluator unit tests."""

import pytest


def test_ethical_evaluator_safe_text():
    pytest.importorskip("detoxify")
    pytest.importorskip("transformers")
    from bforbuntyai import EthicalEvaluator

    ev = EthicalEvaluator()
    result = ev.evaluate("The weather is nice today.")
    assert "verdict" in result
    assert "toxicity" in result
    assert "bias" in result
    assert result["verdict"] in ("SAFE", "POTENTIALLY HARMFUL")


def test_ethical_evaluator_batch():
    pytest.importorskip("detoxify")
    pytest.importorskip("transformers")
    from bforbuntyai import EthicalEvaluator

    ev = EthicalEvaluator()
    results = ev.evaluate_batch(["Hello world.", "Nice day."])
    assert len(results) == 2
    for r in results:
        assert "verdict" in r


def test_text_generator_imports():
    pytest.importorskip("transformers")
    # Just test that the class can be imported; full generation requires model download
    from bforbuntyai import TextGenerator

    assert TextGenerator is not None


def test_text_finetuner_imports():
    pytest.importorskip("transformers")
    from bforbuntyai import TextFineTuner

    assert TextFineTuner is not None


def test_image_captioner_imports():
    pytest.importorskip("transformers")
    from bforbuntyai import ImageCaptioner

    assert ImageCaptioner is not None


def test_stable_diffusion_imports():
    pytest.importorskip("diffusers")
    from bforbuntyai import StableDiffusion

    assert StableDiffusion is not None
