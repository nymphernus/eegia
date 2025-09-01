import hashlib
import numpy as np
from typing import Optional
from storage.database import EEGDatabase
from .sample import EEGSample

class DataManager:
    def __init__(self):
        self.db = EEGDatabase()
    
    def add_sample(self, sample: EEGSample, filename: str) -> str:
        file_hash = self._compute_hash(sample.data)
        
        existing_id = self.db.dataset_exists(file_hash)
        if existing_id:
            return existing_id
        
        dataset_id = self.db.add_dataset(
            filename=filename,
            file_hash=file_hash,
            sfreq=sample.sfreq,
            n_channels=sample.data.shape[0],
            n_samples=sample.data.shape[1],
            ch_names=sample.ch_names,
            data=sample.data,
            metadata=sample.metadata
        )
        return dataset_id
    
    def get_sample(self, dataset_id: str) -> Optional[EEGSample]:
        info = self.db.get_dataset_info(dataset_id)
        if not info:
            return None
        
        data = self.db.get_dataset_data(dataset_id)
        if data is None:
            return None
        
        return EEGSample(
            data=data,
            sfreq=info['sfreq'],
            ch_names=info['ch_names'],
            metadata=info['metadata'],
            raw_path=info['filename']
        )
    
    def list_samples(self) -> list:
        return self.db.list_datasets()
    
    def delete_sample(self, dataset_id: str) -> bool:
        return self.db.delete_dataset(dataset_id)
    
    def _compute_hash(self, data: np.ndarray) -> str:
        return hashlib.md5(data.tobytes()).hexdigest()