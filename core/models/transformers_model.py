import numpy as np
from typing import Any, Dict, List
from core.models.base import ModelBase
import os

class TransformersModel(ModelBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.cache_dir = os.path.join("storage", "models", "transformers")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self, model_path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=self.cache_dir)
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str]) -> np.ndarray:
        import torch
        if not self.tokenizer or not self.model:
            raise RuntimeError("Модель не загружена")

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        if not self.model:
            return {"loaded": False}

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        try:
            import torch
            cuda_info = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            if torch.cuda.is_available():
                cuda_info.update({
                    "cuda_current_device": torch.cuda.current_device(),
                    "cuda_device_name": torch.cuda.get_device_name()
                })
        except ImportError:
            cuda_info = {"cuda_available": False}

        return {
            "name": self.name,
            "framework": "HuggingFace Transformers",
            "device": self.device,
            "cache_dir": self.cache_dir,
            "total_params": f"{total:,}",
            "trainable_params": f"{trainable:,}",
            "devices_info": cuda_info,
            "loaded": True
        }