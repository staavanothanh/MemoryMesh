"""Session registry — auto-save and restore session context."""

import json
import uuid
import zlib
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import aiosqlite

logger = logging.getLogger(__name__)


class SessionStore:
    """SQLite-backed session registry + conversation log."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    _REQUIRED_SESSION_COLUMNS = {
        "session_id": "session_id TEXT PRIMARY KEY",
        "user_id": "user_id TEXT NOT NULL",
        "system_prompt": "system_prompt TEXT DEFAULT ''",
        "workspace_path": "workspace_path TEXT DEFAULT ''",
        "status": "status TEXT NOT NULL DEFAULT 'active'",
        "created_at": "created_at TEXT NOT NULL",
        "updated_at": "updated_at TEXT NOT NULL",
        "ended_at": "ended_at TEXT DEFAULT NULL",
        "deleted": "deleted INTEGER DEFAULT 0",
    }

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                system_prompt TEXT DEFAULT '',
                workspace_path TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT DEFAULT NULL
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS context_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT DEFAULT '',
                tool_args TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS workspace_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                snapshot_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS raw_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                input_payload BLOB,
                output_payload BLOB,
                is_compressed INTEGER DEFAULT 1,
                execution_time_ms REAL,
                status TEXT CHECK(status IN ('success', 'error'))
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_log_session ON context_log(session_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_log_session_time ON raw_log(session_id, timestamp)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_log_tool ON raw_log(tool_name, timestamp)"
        )
        await self._migrate_sessions_schema()
        await self._db.commit()

    async def _migrate_sessions_schema(self):
        """Add missing columns to the sessions table if schema has changed."""
        cursor = await self._db.execute("PRAGMA table_info(sessions)")
        existing = {row["name"] for row in await cursor.fetchall()}
        to_add = []
        for col_name, col_def in self._REQUIRED_SESSION_COLUMNS.items():
            if col_name not in existing:
                to_add.append(f"ALTER TABLE sessions ADD COLUMN {col_def}")
        for stmt in to_add:
            try:
                await self._db.execute(stmt)
                logger.info("Schema migration: %s", stmt)
            except Exception as e:
                logger.warning("Schema migration failed for %s: %s", stmt, e)

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def _close_stale_sessions(self, user_id: str, stale_minutes: int = 0):
        now = datetime.now(timezone.utc).isoformat()
        if stale_minutes > 0:
            threshold_s = stale_minutes * 60
            await self._db.execute(
                """UPDATE sessions SET status = 'ended', ended_at = ?, updated_at = ?
                   WHERE user_id = ? AND status = 'active'
                     AND (strftime('%s','now') - strftime('%s',updated_at)) > ?""",
                (now, now, user_id, threshold_s),
            )
        else:
            await self._db.execute(
                "UPDATE sessions SET status = 'ended', ended_at = ?, updated_at = ? WHERE user_id = ? AND status = 'active'",
                (now, now, user_id),
            )
        await self._db.commit()

    async def create_session(self, user_id: str, system_prompt: str = "", workspace_path: str = "", auto_close_stale: bool = True, stale_minutes: int = 0) -> str:
        if not user_id:
            from ..config import AppConfig
            user_id = AppConfig.from_env().default_user_id
            logger.warning("create_session called with empty user_id, falling back to default: %s", user_id)
        if auto_close_stale:
            await self._close_stale_sessions(user_id, stale_minutes)
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO sessions (session_id, user_id, system_prompt, workspace_path, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'active', ?, ?)",
            (session_id, user_id, system_prompt, workspace_path, now, now),
        )
        await self._db.commit()
        logger.info("Session created: %s for user %s", session_id, user_id)
        return session_id

    async def end_session(self, session_id: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE sessions SET status = 'ended', ended_at = ?, updated_at = ? WHERE session_id = ?",
            (now, now, session_id),
        )
        await self._db.commit()
        logger.info("Session ended: %s", session_id)

    async def mark_deleted(self, session_id: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE sessions SET status = 'ended', deleted = 1, ended_at = ?, updated_at = ? WHERE session_id = ?",
            (now, now, session_id),
        )
        await self._db.commit()
        logger.info("Session soft-deleted: %s", session_id)

    async def hard_delete_session(self, session_id: str):
        """Permanently delete a session and all associated data (context_log, snapshots)."""
        await self._db.execute("DELETE FROM context_log WHERE session_id = ?", (session_id,))
        await self._db.execute("DELETE FROM workspace_snapshots WHERE session_id = ?", (session_id,))
        await self._db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._db.commit()
        logger.info("Session hard-deleted: %s", session_id)

    async def update_system_prompt(self, session_id: str, system_prompt: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE sessions SET system_prompt = ?, updated_at = ? WHERE session_id = ?",
            (system_prompt, now, session_id),
        )
        await self._db.commit()

    async def log_context(self, session_id: str, role: str, content: str, tool_name: str = "", tool_args: str = ""):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO context_log (session_id, role, content, tool_name, tool_args, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, tool_name, tool_args, now),
        )
        await self._db.commit()

    async def save_workspace_snapshot(self, session_id: str, snapshot_data: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO workspace_snapshots (session_id, snapshot_data, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(snapshot_data), now),
        )
        await self._db.commit()

    async def get_workspace_snapshots(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM workspace_snapshots WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["snapshot_data"] = json.loads(d["snapshot_data"])
            result.append(d)
        return result

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_sessions(self, user_id: str, limit: int = 10, status: Optional[str] = None, include_deleted: bool = False) -> List[Dict[str, Any]]:
        conditions = ["user_id = ?"]
        params: list = [user_id]
        if not include_deleted:
            conditions.append("(deleted IS NULL OR deleted = 0)")
        if status:
            conditions.append("status = ?")
            params.append(status)
        cursor = await self._db.execute(
            f"SELECT * FROM sessions WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_active_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_context_log(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM context_log WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_context_log_count(self, session_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM context_log WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def get_raw_log(self, session_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT id, timestamp, session_id, tool_name, is_compressed, execution_time_ms, status FROM raw_log WHERE session_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_raw_log_detail(self, log_id: int) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM raw_log WHERE id = ?", (log_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("is_compressed") and d.get("input_payload"):
            try:
                d["input_payload"] = zlib.decompress(d["input_payload"]).decode("utf-8")
            except Exception:
                pass
        if d.get("is_compressed") and d.get("output_payload"):
            try:
                d["output_payload"] = zlib.decompress(d["output_payload"]).decode("utf-8")
            except Exception:
                pass
        return d

    async def search_raw_log(self, tool_name: str = "", status: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = " AND ".join(conditions) if conditions else "1=1"
        cursor = await self._db.execute(
            f"SELECT id, timestamp, session_id, tool_name, is_compressed, execution_time_ms, status FROM raw_log WHERE {where} ORDER BY id DESC LIMIT ?",
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
