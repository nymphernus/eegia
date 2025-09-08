import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict
from core.models.base import ModelBase


class EEGNet(nn.Module):
    """Упрощённая версия EEGNet"""
    def __init__(self, nb_classes: int = 2, Chans: int = 64, Samples: int = 128, dropoutRate: float = 0.5):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        self.classify = nn.Linear(32 * (Samples // 4), nb_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)


class EEGNetModel(ModelBase):
    def __init__(self, name: str, nb_classes: int = 2, chans: int = 64, samples: int = 128):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples).to(self.device)
        self.model_path = None

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model_path = model_path

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(data, dtype=torch.float32).to(self.device)
            # Ожидаем вход: (batch, 1, chans, samples)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            outputs = self.model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def get_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.name,
            "framework": "PyTorch (EEGNet)",
            "model_path": self.model_path,
            "device": self.device,
            "total_params": total_params,
            "loaded": self.model is not None,
        }
