from typing import List, Optional

from .._base import BaseModel
from ..auth import get_token


class TextFineTuner(BaseModel):
    """Fine-tune GPT-2 (or GPT-Neo) on a custom dataset using HuggingFace Trainer.

    Usage:
        from bforbuntyai import TextFineTuner, dataset
        data = dataset.HuggingFace("amazon_polarity", split="train[:200]")
        tuner = TextFineTuner(dataset=data)
        tuner.train(epochs=3)
        tuner.generate("This product is")
        tuner.save("./my-gpt2")
    """

    def __init__(
        self,
        dataset,
        model_name: str = "gpt2",
        token: Optional[str] = None,
    ):
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "TextFineTuner requires transformers.\n"
                "Install with: pip install bforbuntyai[transformers]"
            )

        self.dataset = dataset
        self.model_name = model_name
        self.token = get_token(token)

        load_kwargs: dict = {}
        if self.token:
            load_kwargs["token"] = self.token

        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, **load_kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = GPT2LMHeadModel.from_pretrained(model_name, **load_kwargs)
        self._output_dir = "./finetuned-model"

    def train(
        self,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
        output_dir: str = "./finetuned-model",
        max_length: int = 256,
    ) -> "TextFineTuner":
        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        self._output_dir = output_dir
        hf_ds = self.dataset.hf_dataset
        col = self.dataset.text_column

        def _tokenize(batch):
            return self.tokenizer(
                batch[col],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized = hf_ds.map(_tokenize, batched=True, remove_columns=hf_ds.column_names)
        tokenized = tokenized.train_test_split(test_size=0.1)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
            report_to="none",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            data_collator=collator,
        )
        trainer.train()
        self._trainer = trainer
        return self

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        num_return: int = 1,
    ) -> List[str]:
        import torch

        device = next(self._model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        for i, t in enumerate(texts, 1):
            print(f"--- Sample {i} ---\n{t}\n")
        return texts

    def visualize(self, prompt: str = "The product is", **kwargs) -> None:
        self.generate(prompt, **kwargs)

    def save(self, path: str = "./finetuned-model") -> None:
        self._model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load(self, path: str = "./finetuned-model") -> "TextFineTuner":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self._model = GPT2LMHeadModel.from_pretrained(path)
        return self
