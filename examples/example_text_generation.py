from bforbuntyai import TextGenerator, TextFineTuner, dataset, auth

# --- GPT-2 inference (no training) ---
gen = TextGenerator(model_name="gpt2")
gen.generate("Once upon a time", max_tokens=100, num_return=3)
gen.generate("The future of artificial intelligence", temperature=0.8, top_p=0.95)

# --- Fine-tune GPT-2 on a HuggingFace dataset ---
# Optional: login if using a gated model or private dataset
# auth.login()

data = dataset.HuggingFace("amazon_polarity", split="train[:200]")
tuner = TextFineTuner(dataset=data, model_name="gpt2")
tuner.train(epochs=3, batch_size=8, lr=5e-5)
tuner.generate("This product is", max_tokens=80)
tuner.save("./gpt2-finetuned")
