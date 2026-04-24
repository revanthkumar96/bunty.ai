from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def train(self, epochs: int = 10, **kwargs) -> "BaseModel": ...

    @abstractmethod
    def generate(self, **kwargs) -> Any: ...

    @abstractmethod
    def visualize(self, **kwargs) -> None: ...

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support save()")

    def load(self, path: str) -> "BaseModel":
        raise NotImplementedError(f"{self.__class__.__name__} does not support load()")
