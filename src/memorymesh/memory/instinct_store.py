"""InstinctStore — SQLite-backed persistent rules for self-learning behavior."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import aiosqlite

logger = logging.getLogger(__name__)


class InstinctStore:
    """Stores and retrieves learned instinct rules."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS instincts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                condition_json TEXT NOT NULL,
                action_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                trigger_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_instincts_user_active
            ON instincts(user_id, active)
        """)
        await self._db.commit()
        logger.info("InstinctStore initialized at %s", self.db_path)

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def add_instinct(
        self,
        user_id: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        confidence: float = 0.5,
    ) -> str:
        instinct_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO instincts
               (id, user_id, condition_json, action_json, confidence, trigger_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
            (instinct_id, user_id, json.dumps(condition), json.dumps(action),
             round(confidence, 4), now, now),
        )
        await self._db.commit()
        logger.debug("Instinct created: %s for user %s", instinct_id, user_id)
        return instinct_id

    async def get_active_instincts(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM instincts WHERE user_id = ? AND active = 1 ORDER BY confidence DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_all_instincts(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM instincts WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def update_confidence(self, instinct_id: str, confidence: float, increment_trigger: bool = True):
        now = datetime.now(timezone.utc).isoformat()
        if increment_trigger:
            await self._db.execute(
                "UPDATE instincts SET confidence = ?, trigger_count = trigger_count + 1, updated_at = ? WHERE id = ?",
                (round(confidence, 4), now, instinct_id),
            )
        else:
            await self._db.execute(
                "UPDATE instincts SET confidence = ?, updated_at = ? WHERE id = ?",
                (round(confidence, 4), now, instinct_id),
            )
        await self._db.commit()

    async def deactivate(self, instinct_id: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE instincts SET active = 0, updated_at = ? WHERE id = ?",
            (now, instinct_id),
        )
        await self._db.commit()

    async def delete(self, instinct_id: str):
        await self._db.execute("DELETE FROM instincts WHERE id = ?", (instinct_id,))
        await self._db.commit()

    async def count_active(self, user_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM instincts WHERE user_id = ? AND active = 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    def _row_to_dict(self, row: aiosqlite.Row) -> Dict[str, Any]:
        d = dict(row)
        d["condition"] = json.loads(d.pop("condition_json"))
        d["action"] = json.loads(d.pop("action_json"))
        d["active"] = bool(d["active"])
        return d
