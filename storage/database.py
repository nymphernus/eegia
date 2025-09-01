import sqlite3
import json
import os
import uuid
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime

class EEGDatabase:
    def __init__(self, db_path: str = "storage/raw.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.data_dir = "storage/raw_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE,
                    sfreq REAL,
                    n_channels INTEGER,
                    n_samples INTEGER,
                    ch_names TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def add_dataset(self, filename: str, file_hash: str, 
                   sfreq: float, n_channels: int, n_samples: int, 
                   ch_names: list, data: np.ndarray, metadata: dict = None) -> str:
        dataset_id = str(uuid.uuid4())
        
        data_file_path = os.path.join(self.data_dir, f"{dataset_id}.npy")
        np.save(data_file_path, data)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO datasets 
                    (id, filename, file_hash, sfreq, n_channels, n_samples, ch_names, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id, filename, file_hash,
                    sfreq, n_channels, n_samples,
                    json.dumps(ch_names),
                    json.dumps(metadata) if metadata else None
                ))
            return dataset_id
        except sqlite3.IntegrityError:
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id FROM datasets WHERE file_hash = ?", (file_hash,))
                row = cursor.fetchone()
                return row[0] if row else None
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'filename': row['filename'],
                    'file_hash': row['file_hash'],
                    'sfreq': row['sfreq'],
                    'n_channels': row['n_channels'],
                    'n_samples': row['n_samples'],
                    'ch_names': json.loads(row['ch_names']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                    'created_at': row['created_at']
                }
        return None
    
    def get_dataset_data(self, dataset_id: str) -> Optional[np.ndarray]:
        data_file_path = os.path.join(self.data_dir, f"{dataset_id}.npy")
        if os.path.exists(data_file_path):
            return np.load(data_file_path)
        return None
    
    def list_datasets(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, filename, sfreq, n_channels, n_samples, file_hash, created_at
                FROM datasets 
                ORDER BY created_at DESC
            """)
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict['created_at']:
                    try:
                        dt = datetime.fromisoformat(row_dict['created_at'].replace('Z', '+00:00'))
                        row_dict['created_at_formatted'] = dt.strftime('%d.%m.%Y %H:%M')
                    except:
                        row_dict['created_at_formatted'] = row_dict['created_at']
                results.append(row_dict)
            return results
    
    def delete_dataset(self, dataset_id: str) -> bool:
        data_file_path = os.path.join(self.data_dir, f"{dataset_id}.npy")
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            return result.rowcount > 0
    
    def dataset_exists(self, file_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM datasets WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return row[0] if row else None