from typing import Optional

import numpy as np

from .._base import BaseModel
from .._logging import get_logger
from .._utils import plot_grid

_logger = get_logger("models.autoencoder")


class AutoEncoder(BaseModel):
    """AutoEncoder using TensorFlow/Keras.

    Supports MNIST (28x28 grayscale) and CIFAR-10 (32x32 RGB) out of the box.
    Works with any dataset whose .as_numpy() returns float32 arrays in [0,1].

    Usage:
        from bforbuntyai import AutoEncoder, dataset
        ae = AutoEncoder(dataset.MNIST())
        ae.train(epochs=50)
        ae.visualize()
    """

    def __init__(
        self,
        dataset,
        encoding_dim: int = 64,
        loss: str = "auto",
        optimizer: str = "adam",
    ):
        self.dataset = dataset
        self.encoding_dim = encoding_dim
        self._loss_arg = loss
        self._optimizer = optimizer
        self.model = None
        self._history = None

    def _build(self):
        try:
            from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
            from tensorflow.keras.models import Model
        except ImportError:
            raise ImportError(
                "AutoEncoder requires TensorFlow.\n"
                "Install with: pip install bforbuntyai[tensorflow]"
            )

        input_shape = self.dataset.shape          # (H, W, C)
        flat_dim = int(np.prod(input_shape))
        channels = input_shape[-1]

        loss = self._loss_arg
        if loss == "auto":
            loss = "binary_crossentropy" if channels == 1 else "mse"

        inputs = Input(shape=input_shape)
        x = Flatten()(inputs)
        encoded = Dense(self.encoding_dim, activation="relu")(x)
        decoded = Dense(flat_dim, activation="sigmoid")(encoded)
        outputs = Reshape(input_shape)(decoded)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=self._optimizer, loss=loss)
        self._loss_name = loss

    def train(self, epochs: int = 50, batch_size: int = 256, validation_split: float = 0.1) -> "AutoEncoder":
        self._build()
        x_train, _ = self.dataset.as_numpy("train")
        _logger.info("Training AutoEncoder on %s (%d samples)...", self.dataset.name, len(x_train))
        self._history = self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )
        return self

    def generate(self, n: int = 10, split: str = "test") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .train() before .generate()")
        x, _ = self.dataset.as_numpy(split)
        return self.model.predict(x[:n], verbose=0)

    def visualize(self, n: int = 10) -> None:
        x_orig, _ = self.dataset.as_numpy("test")
        x_orig = x_orig[:n]
        x_recon = self.model.predict(x_orig, verbose=0)
        images = list(x_orig) + list(x_recon)
        titles = [f"Orig {i}" for i in range(n)] + [f"Recon {i}" for i in range(n)]
        plot_grid(images, titles=titles, cols=n)

    def save(self, path: str = "autoencoder.keras") -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        self.model.save(path)
        _logger.info("Saved to %s", path)

    def load(self, path: str = "autoencoder.keras") -> "AutoEncoder":
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError("TensorFlow is required to load an AutoEncoder.")
        self.model = load_model(path)
        return self
