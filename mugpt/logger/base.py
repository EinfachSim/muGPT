from abc import ABC, abstractmethod

class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, float], step: int) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...