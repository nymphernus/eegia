import sqlite3
import json
import os
import uuid
from typing import Optional, Dict, List
from datetime import datetime


class ModelsDatabase:
    def __init__(self, db_path: str = "storage/models.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    file_path TEXT,
                    file_hash TEXT UNIQUE,
                    framework_version TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)")
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_type_path
                ON models(model_type, file_path)
            """)
    
    def add_model(self, name: str, model_type: str, 
                  file_path: str, file_hash: Optional[str],
                  framework_version: str = None,
                  metadata: dict = None) -> str:
        model_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO models 
                    (id, name, model_type, file_path, file_hash, framework_version, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, name, model_type, file_path, file_hash,
                    framework_version, json.dumps(metadata) if metadata else None
                ))
            return model_id
        except sqlite3.IntegrityError:
            with sqlite3.connect(self.db_path) as conn:
                if file_hash:
                    cursor = conn.execute("SELECT id FROM models WHERE file_hash = ?", (file_hash,))
                    row = cursor.fetchone()
                    if row:
                        return row[0]
                cursor = conn.execute(
                    "SELECT id FROM models WHERE model_type = ? AND file_path = ?",
                    (model_type, file_path)
                )
                row = cursor.fetchone()
                return row[0] if row else None
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'model_type': row['model_type'],
                    'file_path': row['file_path'],
                    'file_hash': row['file_hash'],
                    'framework_version': row['framework_version'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                    'created_at': row['created_at']
                }
        return None
    
    def list_models(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, name, model_type, file_path, file_hash, created_at
                FROM models 
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
    
    def delete_model(self, model_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            return result.rowcount > 0
    
    def model_exists(self, file_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM models WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return row[0] if row else None

    def model_exists_by_path(self, model_type: str, file_path: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM models WHERE model_type = ? AND file_path = ?",
                (model_type, file_path)
            )
            row = cursor.fetchone()
            return row[0] if row else None
