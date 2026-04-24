from bforbuntyai import EthicalEvaluator

ev = EthicalEvaluator()

# Single text
ev.evaluate("The weather is lovely today.")
ev.evaluate("I hate everyone in this room.")

# Batch evaluation
texts = [
    "AI will transform healthcare for the better.",
    "This group of people is inferior.",
    "Let's build a more inclusive world.",
]
results = ev.evaluate_batch(texts)

# Inspect programmatically
for r in results:
    print(f"[{r['verdict']}] {r['text'][:50]}")
