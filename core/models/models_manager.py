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

    def _hash_file(self, file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _derive_name(self, file_path_or_repo: str) -> str:
        base = os.path.basename(file_path_or_repo)
        name, _ = os.path.splitext(base)
        return name

    def _persist_local_file(self, src_path: str, model_type: str) -> str:
        dst_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(dst_dir, exist_ok=True)

        base = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, base)

        if os.path.exists(dst_path):
            name, ext = os.path.splitext(base)
            i = 1
            while True:
                cand = os.path.join(dst_dir, f"{name}_{i}{ext}")
                if not os.path.exists(cand):
                    dst_path = cand
                    break
                i += 1

        shutil.copy2(src_path, dst_path)
        return dst_path

    def add_model(
        self,
        name: Optional[str],
        model_type: str,
        file_path: str,
        framework_version: str = "?",
        metadata: Optional[Dict] = None,
        assume_hf_if_missing: bool = True
    ) -> str:
        if not name or not name.strip():
            name = self._derive_name(file_path)

        file_hash: Optional[str] = None
        stored_path = file_path

        if os.path.exists(file_path):
            file_hash = self._hash_file(file_path)
            existing_id = self.db.model_exists(file_hash)
            if existing_id:
                return existing_id
            stored_path = self._persist_local_file(file_path, model_type)
        else:
            if not (assume_hf_if_missing and model_type == "transformers"):
                raise FileNotFoundError(f"Файл модели не найден: {file_path}")
            existing_id = self.db.model_exists_by_path(model_type, file_path)
            if existing_id:
                return existing_id

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
            return None

        model = get_model(info["model_type"], info["name"])

        if info["model_type"] == "transformers":
            model.load(info["file_path"])
        else:
            if not os.path.exists(info["file_path"]):
                raise FileNotFoundError(f"Файл модели не найден: {info['file_path']}")
            model.load(info["file_path"])

        return model

    def list_models(self) -> List[Dict]:
        return self.db.list_models()

    def delete_model(self, model_id: str) -> bool:
        info = self.db.get_model_info(model_id)
        ok = self.db.delete_model(model_id)
        if ok and info and info["file_path"]:
            try:
                if info["file_path"].startswith(self.models_dir) and os.path.exists(info["file_path"]):
                    os.remove(info["file_path"])
            except Exception:
                pass
        return ok

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        return self.db.get_model_info(model_id)
