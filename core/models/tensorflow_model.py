import numpy as np
from typing import Any, Dict
from core.models.base import ModelBase

class TensorFlowModel(ModelBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.device = "cpu"
        self.model = None

    def load(self, model_path: str):
        import tensorflow as tf
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(model_path)

    def predict(self, data: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        with tf.device(self.device):
            output = self.model.predict(data, verbose=0)
        return np.array(output)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "framework": "TensorFlow",
            "device": self.device,
            "loaded": self.model is not None
        }
