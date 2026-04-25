"""TDD tests: universal model support (any HuggingFace model, not hardcoded ones).

These tests verify that TextGenerator, TextFineTuner, ImageCaptioner, and
StableDiffusion work with ANY compatible HuggingFace model, not just the
hardcoded defaults (gpt2, blip-base, stable-diffusion-v1-5).

Most tests are mocked so they run without downloading weights.
Tests that require transformers/diffusers are skipped automatically when
those packages are not installed or have a version conflict.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_text_generator_uses_auto_tokenizer():
    """TextGenerator must use AutoTokenizer, not GPT2Tokenizer."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_tok = MagicMock()
    mock_tok.eos_token = "<eos>"
    mock_tok.eos_token_id = 0
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as tok_cls, \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        from bforbuntyai import TextGenerator

        gen = TextGenerator(model_name="EleutherAI/gpt-neo-125M")
        assert gen.model_name == "EleutherAI/gpt-neo-125M"
        tok_cls.assert_called()


def test_text_generator_uses_auto_model():
    """TextGenerator must use AutoModelForCausalLM, not GPT2LMHeadModel."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_tok = MagicMock()
    mock_tok.eos_token = "<eos>"
    mock_tok.eos_token_id = 0
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mdl_cls:
        from bforbuntyai import TextGenerator

        TextGenerator(model_name="mistralai/Mistral-7B-v0.1")
        mdl_cls.assert_called()


def test_text_finetuner_uses_auto_classes():
    """TextFineTuner must use Auto classes so any causal-LM can be fine-tuned."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_tok = MagicMock()
    mock_tok.eos_token = "<eos>"
    mock_tok.eos_token_id = 0
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.hf_dataset = MagicMock()
    mock_ds.text_column = "text"

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as tok_cls, \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mdl_cls:
        from bforbuntyai import TextFineTuner

        tuner = TextFineTuner(dataset=mock_ds, model_name="EleutherAI/gpt-neo-125M")
        assert tuner.model_name == "EleutherAI/gpt-neo-125M"
        tok_cls.assert_called()
        mdl_cls.assert_called()


def test_text_finetuner_load_uses_auto_classes():
    """TextFineTuner.load() must use Auto classes so any saved model can be reloaded."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_tok = MagicMock()
    mock_tok.eos_token = "<eos>"
    mock_tok.eos_token_id = 0
    mock_model = MagicMock()
    mock_ds = MagicMock()
    mock_ds.hf_dataset = MagicMock()
    mock_ds.text_column = "text"

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mdl_cls:
        from bforbuntyai import TextFineTuner

        tuner = TextFineTuner(dataset=mock_ds)
        tuner.load("/some/saved/path")
        # from_pretrained called at least twice: __init__ + load()
        assert mdl_cls.call_count >= 2


def test_image_captioner_uses_pipeline():
    """ImageCaptioner must use pipeline('image-to-text') to support any captioning model."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_pipe = MagicMock()

    with patch("transformers.pipeline", return_value=mock_pipe) as pipe_fn:
        from bforbuntyai import ImageCaptioner

        cap = ImageCaptioner(model_name="nlpconnect/vit-gpt2-image-captioning")
        assert cap.model_name == "nlpconnect/vit-gpt2-image-captioning"
        pipe_fn.assert_called_once()
        assert pipe_fn.call_args[0][0] == "image-to-text"


def test_image_captioner_caption_calls_pipeline():
    """ImageCaptioner.caption() must call the pipeline and return generated text."""
    pytest.importorskip("transformers", exc_type=ImportError)
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "a photo of a cat"}]

    with patch("transformers.pipeline", return_value=mock_pipe):
        from bforbuntyai import ImageCaptioner

        cap = ImageCaptioner()
        fake_image = np.zeros((64, 64, 3), dtype=np.float32)
        text = cap.caption(fake_image)
        assert text == "a photo of a cat"
        mock_pipe.assert_called_once()


def test_stable_diffusion_uses_auto_pipeline():
    """StableDiffusion must use AutoPipelineForText2Image for universal SD model support."""
    pytest.importorskip("diffusers", exc_type=ImportError)
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe

    with patch("diffusers.AutoPipelineForText2Image.from_pretrained", return_value=mock_pipe) as auto_fn:
        from bforbuntyai import StableDiffusion

        sd = StableDiffusion(model_id="stabilityai/stable-diffusion-xl-base-1.0")
        assert sd.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        auto_fn.assert_called_once()
