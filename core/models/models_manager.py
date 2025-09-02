import hashlib
import os
import shutil
from typing import Optional, Dict, List
from storage.models_database import ModelsDatabase
from core.models.registry import get_model

class ModelsManager:
    def __init__(self):
        self.db = ModelsDatabase()
        self.models_dir = "storage/models"
        os.makedirs(self.models_dir, exist_ok=True)

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _derive_name(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _persist_file(self, src: str, model_type: str) -> str:
        dst_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))

        if os.path.exists(dst):
            name, ext = os.path.splitext(os.path.basename(src))
            i = 1
            while os.path.exists(os.path.join(dst_dir, f"{name}_{i}{ext}")):
                i += 1
            dst = os.path.join(dst_dir, f"{name}_{i}{ext}")

        shutil.copy2(src, dst)
        return dst

    def add_model(
        self,
        name: Optional[str],
        model_type: str,
        file_path: str,
        framework_version: str = "?",
        metadata: Optional[Dict] = None
    ) -> str:
        name = name or self._derive_name(file_path)
        file_hash = None
        stored_path = file_path

        if os.path.exists(file_path):
            file_hash = self._hash_file(file_path)
            if (existing := self.db.model_exists(file_hash)):
                return existing
            stored_path = self._persist_file(file_path, model_type)
        elif model_type != "transformers":
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if existing := self.db.model_exists_by_path(model_type, file_path):
            return existing

        return self.db.add_model(
            name=name,
            model_type=model_type,
            file_path=stored_path,
            file_hash=file_hash,
            framework_version=framework_version,
            metadata=metadata
        )

    def load_model(self, model_id: str):
        info = self.db.get_model_info(model_id)
        if not info:
            raise ValueError("Модель не найдена")

        model = get_model(info["model_type"], info["name"])
        model.load(info["file_path"])
        return model

    def list_models(self) -> List[Dict]:
        return self.db.list_models()

    def delete_model(self, model_id: str) -> bool:
        info = self.db.get_model_info(model_id)
        ok = self.db.delete_model(model_id)
        if ok and info and os.path.exists(info["file_path"]):
            try:
                os.remove(info["file_path"])
            except:
                pass
        return ok

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        return self.db.get_model_info(model_id)