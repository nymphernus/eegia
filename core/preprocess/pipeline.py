import yaml, json
import numpy as np
from typing import List, Dict, Any, Union
from core.preprocess.base import Transform
from core.preprocess.steps.filters import BandpassFilter, NotchFilter
from core.preprocess.steps.resample_normalize import Resample, Normalize
from core.preprocess.steps.artifacts import ReReference, ICAFilter, Epoching

STEP_REGISTRY = {
    "bandpass": BandpassFilter,
    "notch": NotchFilter,
    "resample": Resample,
    "normalize": Normalize,
    "rereference": ReReference,
    "ica": ICAFilter,
    "epoching": Epoching,
}

class PreprocessPipeline:
    def __init__(self, steps: List[Transform] = None, random_state: int = 42):
        self.steps = steps or []
        self.random_state = random_state

    def add_step(self, step: Transform):
        self.steps.append(step)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def to_dict(self):
        return {
            "random_state": self.random_state,
            "steps": [s.to_dict() for s in self.steps]
        }

    def save(self, path: str):
        with open(path, "w") as f:
            if path.endswith((".yaml", ".yml")):
                yaml.safe_dump(self.to_dict(), f)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PreprocessPipeline":
            with open(path, "r") as f:
                if path.endswith((".yaml", ".yml")):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            steps = []
            for s in config.get("steps", []):
                step_name = s["name"]
                params = s.get("params", {})
                if step_name not in STEP_REGISTRY:
                    raise ValueError(f"Unknown step: {step_name}")
                steps.append(STEP_REGISTRY[step_name](**params))

            return cls(steps=steps, random_state=config.get("random_state", 42))