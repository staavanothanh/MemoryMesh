"""Session registry — auto-save and restore session context."""

import uuid
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

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                system_prompt TEXT DEFAULT '',
                workspace_path TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def create_session(self, user_id: str, system_prompt: str = "", workspace_path: str = "") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO sessions (session_id, user_id, system_prompt, workspace_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, system_prompt, workspace_path, now, now),
        )
        await self._db.commit()
        logger.info("Session created: %s for user %s", session_id, user_id)
        return session_id

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

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_context_log(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM context_log WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
