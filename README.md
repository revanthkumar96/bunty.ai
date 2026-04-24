# bforbuntyai

Run GenAI experiments in a few lines instead of hundreds.

```python
from bforbuntyai import GAN, dataset
gan = GAN(dataset.FashionMNIST())
gan.train(epochs=50)
gan.generate(n=25)
```

## Installation

```bash
# Core (no heavy ML libs)
pip install bforbuntyai

# With PyTorch (GAN, DCGAN, cGAN, VAE, Pix2Pix)
pip install "bforbuntyai[torch]"

# With TensorFlow (AutoEncoder)
pip install "bforbuntyai[tensorflow]"

# With HuggingFace (TextGenerator, TextFineTuner, ImageCaptioner, EthicalEvaluator)
pip install "bforbuntyai[transformers]"

# With Stable Diffusion
pip install "bforbuntyai[diffusers]"

# With ethical evaluation
pip install "bforbuntyai[ethics]"

# Everything at once
pip install "bforbuntyai[all]"
```

## HuggingFace Authentication

Some models (Stable Diffusion gated versions, private datasets) require a HuggingFace token.

```python
from bforbuntyai import auth

auth.login()                    # interactive browser prompt
auth.login(token="hf_...")      # explicit token
auth.whoami()                   # check who is logged in
auth.logout()

# Or pass token directly to any model
sd = StableDiffusion(token="hf_...")
data = dataset.HuggingFace("private/dataset", token="hf_...")
```

Token is also auto-read from the `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable.

## Models & Datasets

### AutoEncoder (TensorFlow)
```python
from bforbuntyai import AutoEncoder, dataset

ae = AutoEncoder(dataset.MNIST(), encoding_dim=64)
ae.train(epochs=50)
ae.visualize()              # original vs reconstructed grid
ae.save("ae.keras")
```

### GAN — Vanilla (PyTorch)
```python
from bforbuntyai import GAN, dataset

gan = GAN(dataset.FashionMNIST(), latent_dim=100)
gan.train(epochs=50)
gan.generate(n=25)
gan.save("gan.pth")
```

### DCGAN — Convolutional GAN (PyTorch)
```python
from bforbuntyai import DCGAN, dataset

dcgan = DCGAN(dataset.MNIST())
dcgan.train(epochs=20)
dcgan.generate(n=25)
```

### Conditional GAN (PyTorch)
```python
from bforbuntyai import ConditionalGAN, dataset

cgan = ConditionalGAN(dataset.MNIST(), num_classes=10)
cgan.train(epochs=50)
cgan.generate_class(labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### VAE — Variational AutoEncoder (PyTorch)
```python
from bforbuntyai import VAE, dataset

vae = VAE(dataset.MNIST(), latent_dim=2)
vae.train(epochs=20)
vae.visualize()              # reconstructions
vae.visualize_latent()       # 2-D scatter plot of latent space
vae.interpolate(img_a, img_b, steps=10)    # morph between images
```

### Pix2Pix Image Translation (PyTorch)
```python
from bforbuntyai import Pix2Pix, dataset

p2p = Pix2Pix(dataset.Edges2Shoes())      # auto-downloads ~1.5 GB
p2p.train(epochs=5)
p2p.visualize()              # input edge | generated photo | real photo
```

### GPT-2 Text Generation
```python
from bforbuntyai import TextGenerator

gen = TextGenerator(model_name="gpt2")
gen.generate("Once upon a time", max_tokens=100, num_return=3)
```

### GPT-2 Fine-tuning
```python
from bforbuntyai import TextFineTuner, dataset

data = dataset.HuggingFace("amazon_polarity", split="train[:200]")
tuner = TextFineTuner(dataset=data)
tuner.train(epochs=3)
tuner.generate("This product is")
tuner.save("./my-gpt2")
```

### Stable Diffusion
```python
from bforbuntyai import StableDiffusion, auth

auth.login()    # required for some model versions
sd = StableDiffusion()
sd.generate("A sunset over mountains, 4K photo", n=4, steps=20)
```

### BLIP Image Captioning
```python
from bforbuntyai import ImageCaptioner

cap = ImageCaptioner()
cap.caption("photo.jpg")
cap.caption("https://example.com/image.jpg")
cap.visualize(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### Ethical Evaluation
```python
from bforbuntyai import EthicalEvaluator

ev = EthicalEvaluator()
ev.evaluate("Some text to check for toxicity and bias")
ev.evaluate_batch(["text one", "text two", "text three"])
```

## Supported Datasets

| Dataset | Class | Used by |
|---|---|---|
| MNIST | `dataset.MNIST()` | AutoEncoder, GAN, DCGAN, cGAN, VAE |
| Fashion-MNIST | `dataset.FashionMNIST()` | GAN, DCGAN, VAE |
| CIFAR-10 | `dataset.CIFAR10()` | AutoEncoder |
| HuggingFace | `dataset.HuggingFace("name", split="...")` | TextFineTuner |
| Edges2Shoes | `dataset.Edges2Shoes()` | Pix2Pix |
| Custom folder | `dataset.Custom("path/", image_size=64)` | Any image model |

## Common API

Every model shares the same interface:

| Method | Description |
|---|---|
| `.train(epochs=...)` | Train the model, returns `self` for chaining |
| `.generate(n=...)` | Generate samples |
| `.visualize()` | Plot results with matplotlib |
| `.save(path)` | Save model weights |
| `.load(path)` | Load model weights |

## Cache

Downloaded datasets and models are cached at `~/.bforbuntyai/cache/`.
