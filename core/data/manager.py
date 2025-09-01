import uuid
import json
from typing import Dict, List
from .sample import EEGSample


class DataManager:
    def __init__(self, catalog_path: str = "storage/datasets.json"):
        self.catalog_path = catalog_path
        self.datasets: Dict[str, EEGSample] = {}

    def add_sample(self, sample: EEGSample) -> str:
        sample_id = str(uuid.uuid4())
        self.datasets[sample_id] = sample
        self._save_catalog()
        return sample_id

    def get_sample(self, sample_id: str) -> EEGSample:
        return self.datasets[sample_id]

    def list_samples(self) -> List[Dict]:
        return [
            {"id": sid, "subject": s.subject_id, "task": s.task, "path": s.raw_path}
            for sid, s in self.datasets.items()
        ]

    def _save_catalog(self):
        catalog = [
            {
                "id": sid,
                "subject": s.subject_id,
                "task": s.task,
                "sfreq": s.sfreq,
                "n_channels": s.data.shape[0],
                "path": s.raw_path,
            }
            for sid, s in self.datasets.items()
        ]
        with open(self.catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
