import re
import logging
from typing import List, Dict, Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)


def sanitize_fts_query(query: str) -> str:
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    return " ".join(cleaned.split())


class FTSBackend:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        cursor = await self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        )
        row = await cursor.fetchone()
        if row and "level" not in row["sql"]:
            await self._db.execute("DROP TABLE IF EXISTS memory_fts")
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                memory_id UNINDEXED,
                user_id UNINDEXED,
                level UNINDEXED,
                content,
                tokenize='unicode61'
            )
        """)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def add(self, memory_id: str, content: str, user_id: str, level: str = "user"):
        await self._db.execute(
            "INSERT INTO memory_fts (memory_id, user_id, level, content) VALUES (?, ?, ?, ?)",
            (memory_id, user_id, level, content),
        )
        await self._db.commit()

    async def search(
        self, query_text: str, user_id: str, limit: int = 10,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        cleaned = sanitize_fts_query(query_text)
        if not cleaned:
            return []
        if level_filter:
            placeholders = ",".join("?" for _ in level_filter)
            sql = f"""
                SELECT memory_id, content, rank
                FROM memory_fts
                WHERE user_id = ? AND level IN ({placeholders}) AND content MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            params = (user_id, *level_filter, cleaned, limit)
        else:
            sql = """
                SELECT memory_id, content, rank
                FROM memory_fts
                WHERE user_id = ? AND content MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            params = (user_id, cleaned, limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {"id": row["memory_id"], "content": row["content"], "score": row["rank"]}
            for row in rows
        ]

    async def update(self, memory_id: str, content: str) -> bool:
        """Update content for an existing FTS entry."""
        try:
            await self._db.execute(
                "UPDATE memory_fts SET content = ? WHERE memory_id = ?",
                (content, memory_id),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.warning("FTS update failed for %s: %s", memory_id, e)
            return False

    async def delete(self, memory_id: str) -> bool:
        try:
            await self._db.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.warning("FTS delete failed for %s: %s", memory_id, e)
            return False

    async def get_all_ids(self) -> List[str]:
        cursor = await self._db.execute("SELECT memory_id FROM memory_fts")
        rows = await cursor.fetchall()
        return [row["memory_id"] for row in rows]

    async def list_recent(
        self, user_id: str, limit: int = 10,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Chronological fallback — sort by insert order (rowid DESC)."""
        if level_filter:
            placeholders = ",".join("?" for _ in level_filter)
            sql = f"""
                SELECT memory_id, content, rowid
                FROM memory_fts
                WHERE user_id = ? AND level IN ({placeholders})
                ORDER BY rowid DESC
                LIMIT ?
            """
            params = (user_id, *level_filter, limit)
        else:
            sql = """
                SELECT memory_id, content, rowid
                FROM memory_fts
                WHERE user_id = ?
                ORDER BY rowid DESC
                LIMIT ?
            """
            params = (user_id, limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {"id": row["memory_id"], "content": row["content"], "score": 0.0}
            for row in rows
        ]

    async def reindex(self, memory_id: str, content: str, user_id: str, level: str = "user"):
        await self._db.execute(
            "INSERT OR REPLACE INTO memory_fts (memory_id, user_id, level, content) VALUES (?, ?, ?, ?)",
            (memory_id, user_id, level, content),
        )
        await self._db.commit()
