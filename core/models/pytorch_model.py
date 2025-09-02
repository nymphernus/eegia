import numpy as np
from typing import Any, Dict
from core.models.base import ModelBase
import os

class PyTorchModel(ModelBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.device = self._get_best_device()
        self.model = None
        self.cache_dir = os.path.join("models", "pytorch")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_best_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def load(self, model_path: str):
        import torch
        self.model = torch.load(model_path, map_location=self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def predict(self, data: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(data).float().to(self.device)
            output = self.model(tensor)
            return output.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "name": self.name,
            "framework": "PyTorch",
            "device": self.device,
            "total_params": f"{total_params:,}",
            "loaded": self.model is not None
        }