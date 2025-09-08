import os
import joblib
import numpy as np
from typing import Any, Dict
from core.models.base import ModelBase


class LightGBMModel(ModelBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = None
        self.model_path = None

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        return self.model.predict(data)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "framework": "LightGBM",
            "model_path": self.model_path,
            "loaded": self.model is not None,
        }
