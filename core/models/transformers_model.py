import numpy as np
from typing import Any, Dict, List, Union
from core.models.base import ModelBase
import os


class TransformersModel(ModelBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.cache_dir = os.path.join("storage", "models", "transformers")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model_type = None  # text_classification | time_series | unknown

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self, model_path: str):
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, pipeline

        # Сначала получаем конфиг
        config = AutoConfig.from_pretrained(model_path, cache_dir=self.cache_dir)

        # Определяем тип задачи
        if hasattr(config, "architectures") and any("Chronos" in arch for arch in config.architectures):
            self.model_type = "time_series"
            from transformers import AutoModelForTimeSeriesForecasting
            self.model = AutoModelForTimeSeriesForecasting.from_pretrained(
                model_path, cache_dir=self.cache_dir
            ).to(self.device)
            self.pipeline = None
            self.tokenizer = None
        else:
            # По умолчанию считаем text-classification
            self.model_type = "text_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, cache_dir=self.cache_dir
            ).to(self.device)
            self.pipeline = pipeline(
                task="text-classification", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1
            )

        self.model.eval()

    def predict(self, X: Union[List[str], np.ndarray], **kwargs) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Модель не загружена")

        if self.model_type == "text_classification":
            if not isinstance(X, list):
                # Попробуем преобразовать numpy → list[str]
                if isinstance(X, np.ndarray):
                    X = [str(x) for x in X.tolist()]
                else:
                    raise ValueError("Для text_classification ожидается список строк (list[str])")

            # Правильная токенизация → int64 индексы
            inputs = self.tokenizer(
                X,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs).logits

            # Классы
            if outputs.shape[1] > 1:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = torch.sigmoid(outputs).cpu().numpy()

            return preds

        elif self.model_type == "time_series":
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)

            prediction_length = kwargs.get("prediction_length", 10)
            num_samples = kwargs.get("num_samples", 20)

            with torch.no_grad():
                preds = self.model.generate(
                    past_values=torch.tensor(X, dtype=torch.float32).to(self.device),
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )
            return preds.cpu().numpy()

        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")


    def get_info(self) -> Dict[str, Any]:
        if not self.model:
            return {"loaded": False}

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        try:
            import torch
            cuda_info = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            if torch.cuda.is_available():
                cuda_info.update(
                    {
                        "cuda_current_device": torch.cuda.current_device(),
                        "cuda_device_name": torch.cuda.get_device_name(),
                    }
                )
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
            "loaded": True,
            "model_type": self.model_type,
        }
