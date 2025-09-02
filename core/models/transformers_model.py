import numpy as np
from typing import Any, Dict, List
from core.models.base import ModelBase
import os

class TransformersModel(ModelBase):
    def __init__(self, name: str, device: str = "auto"):
        super().__init__(name)
        self.device = self._get_best_device()
        self.model = None
        self.tokenizer = None
        self.cache_dir = os.path.join("models", "transformers")
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

    def load(self, model_name_or_path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"📥 Загрузка модели: {model_name_or_path}")
        print(f"💾 Кэш директория: {self.cache_dir}")
        print(f"🖥  Используемое устройство: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            cache_dir=self.cache_dir
        )
        
        if self.device != "cpu":
            print(f"🔄 Перенос модели на {self.device}...")
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ Модель успешно загружена!")

    def predict(self, texts: List[str]) -> np.ndarray:
        import torch
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Модель не загружена")

        print(f"🔮 Выполняем инференс на {self.device}...")
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        print("✅ Инференс завершён!")
        return logits.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        
        devices_info = {}
        try:
            import torch
            devices_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                devices_info["cuda_devices"] = torch.cuda.device_count()
                devices_info["cuda_current_device"] = torch.cuda.current_device()
                devices_info["cuda_device_name"] = torch.cuda.get_device_name()
        except ImportError:
            devices_info["cuda_available"] = False
        
        return {
            "name": self.name,
            "framework": "HuggingFace Transformers",
            "requested_device": self.requested_device,
            "actual_device": self.device,
            "cache_dir": self.cache_dir,
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "devices_info": devices_info,
            "loaded": self.model is not None
        }