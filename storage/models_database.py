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
            for idx in [
                "CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type)",
                "CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_type_path ON models(model_type, file_path)"
            ]:
                conn.execute(idx)

    def add_model(self, **kwargs) -> str:
        model_id = str(uuid.uuid4())
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO models (id, name, model_type, file_path, file_hash, framework_version, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    kwargs['name'],
                    kwargs['model_type'],
                    kwargs['file_path'],
                    kwargs.get('file_hash'),
                    kwargs.get('framework_version'),
                    json.dumps(kwargs.get('metadata')) if kwargs.get('metadata') else None
                ))
            return model_id
        except sqlite3.IntegrityError:
            with sqlite3.connect(self.db_path) as conn:
                if kwargs.get('file_hash'):
                    cursor = conn.execute("SELECT id FROM models WHERE file_hash = ?", (kwargs['file_hash'],))
                else:
                    cursor = conn.execute(
                        "SELECT id FROM models WHERE model_type = ? AND file_path = ?",
                        (kwargs['model_type'], kwargs['file_path'])
                    )
                row = cursor.fetchone()
                return row[0] if row else None

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if row:
                return {
                    **dict(row),
                    "metadata": json.loads(row['metadata']) if row['metadata'] else None
                }

    def list_models(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, name, model_type, file_path, file_hash, created_at
                FROM models ORDER BY created_at DESC
            """).fetchall()

            result = []
            for row in rows:
                d = dict(row)
                if d['created_at']:
                    try:
                        dt = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                        d['created_at_formatted'] = dt.strftime('%d.%m.%Y %H:%M')
                    except:
                        d['created_at_formatted'] = d['created_at']
                result.append(d)
            return result

    def delete_model(self, model_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("DELETE FROM models WHERE id = ?", (model_id,)).rowcount > 0

    def model_exists(self, file_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT id FROM models WHERE file_hash = ?", (file_hash,)).fetchone()
            return row[0] if row else None

    def model_exists_by_path(self, model_type: str, file_path: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM models WHERE model_type = ? AND file_path = ?",
                (model_type, file_path)
            ).fetchone()
            return row[0] if row else None