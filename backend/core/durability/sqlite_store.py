import sqlite3
import json
from typing import Optional
from core.durability.contracts import DurableStore, Checkpoint, FailureRecord

class SqliteStore(DurableStore):
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # For :memory: databases, we must keep the connection open
        self._mem_conn = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._init_db()

    def _get_connection(self):
        if self._mem_conn:
            return self._mem_conn
        return sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)

    def _init_db(self):
        conn = self._get_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                workflow_id TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT,
                node TEXT,
                error TEXT,
                attempt INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        if not self._mem_conn:
            conn.close()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        conn = self._get_connection()
        try:
            data = checkpoint.model_dump_json()
            conn.execute(
                'INSERT OR REPLACE INTO checkpoints (workflow_id, data) VALUES (?, ?)',
                (checkpoint.workflow_id, data)
            )
        finally:
            if not self._mem_conn:
                conn.close()

    def get_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                'SELECT data FROM checkpoints WHERE workflow_id = ?',
                (workflow_id,)
            )
            row = cursor.fetchone()
            if row:
                return Checkpoint.model_validate_json(row[0])
            return None
        finally:
            if not self._mem_conn:
                conn.close()

    def log_failure(self, workflow_id: str, failure: FailureRecord) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                'INSERT INTO failures (workflow_id, node, error, attempt, timestamp) VALUES (?, ?, ?, ?, ?)',
                (workflow_id, failure.node, failure.error, failure.attempt, failure.timestamp.isoformat())
            )
        finally:
            if not self._mem_conn:
                conn.close()
