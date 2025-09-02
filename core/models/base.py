from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class ModelBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> Any:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass
