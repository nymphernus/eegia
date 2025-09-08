from typing import Dict, Union, Type
from core.models.base import ModelBase

MODEL_REGISTRY: Dict[str, Union[str, Type[ModelBase]]] = {
    "pytorch": "core.models.pytorch_model.PyTorchModel",
    "tensorflow": "core.models.tensorflow_model.TensorFlowModel",
    "transformers": "core.models.transformers_model.TransformersModel",
    "lightgbm": "core.models.lightgbm_model.LightGBMModel",
    "eegnet": "core.models.eegnet_model.EEGNetModel",
}


def get_model(model_type: str, name: str, **kwargs) -> ModelBase:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class_or_path = MODEL_REGISTRY[model_type]

    if isinstance(model_class_or_path, str):
        module_path, class_name = model_class_or_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        MODEL_REGISTRY[model_type] = model_class
    else:
        model_class = model_class_or_path

    return model_class(name, **kwargs)
