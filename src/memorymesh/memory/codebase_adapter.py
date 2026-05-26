"""Anti-Corruption Layer for external codebase-memory-mcp database.
Read-only connection with graceful degradation and timeout."""

import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

import aiosqlite

logger = logging.getLogger(__name__)

CODABASE_CONNECT_TIMEOUT = 2.0
CODABASE_QUERY_TIMEOUT = 1.0


class CodebaseDBAdapter:
    """Read-only adapter for external codebase-memory-mcp database.

    Connects with URI mode=ro (read-only) to avoid locking the external DB.
    Wraps all operations in try/except with timeout for graceful degradation.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._available: bool = False

    async def initialize(self):
        if not self.db_path:
            self._available = False
            return
        try:
            uri = f"file:{self.db_path}?mode=ro"
            self._db = await asyncio.wait_for(
                aiosqlite.connect(uri, uri=True),
                timeout=CODABASE_CONNECT_TIMEOUT,
            )
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA busy_timeout=500")
            await self._db.execute("PRAGMA query_only=ON")
            self._available = True
            logger.info("CodebaseDBAdapter connected (read-only) to %s", self.db_path)
        except asyncio.TimeoutError:
            logger.warning("CodebaseDBAdapter connection timed out to %s", self.db_path)
            self._available = False
        except Exception as e:
            logger.warning("CodebaseDBAdapter init failed: %s", e)
            self._available = False

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._db is not None

    async def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities in external codebase DB — adapted to MemoryMesh schema."""
        if not self.is_available:
            return []
        try:
            cursor = await asyncio.wait_for(
                self._db.execute(
                    """SELECT id, name, type, properties FROM entities
                       WHERE name LIKE ? OR json_extract(properties, '$.summary') LIKE ?
                       ORDER BY updated_at DESC LIMIT ?""",
                    (f"%{query}%", f"%{query}%", limit),
                ),
                timeout=CODABASE_QUERY_TIMEOUT,
            )
            rows = await cursor.fetchall()
            adapted = []
            for r in rows:
                adapted.append({
                    "id": r["id"],
                    "name": r["name"],
                    "type": r.get("type", "concept"),
                    "properties": json.loads(r.get("properties", "{}")),
                    "source": "codebase_memory",
                })
            return adapted
        except asyncio.TimeoutError:
            logger.warning("CodebaseDBAdapter search timed out for query: %s", query)
            return []
        except Exception as e:
            logger.warning("CodebaseDBAdapter search failed: %s", e)
            return []

    async def search_relations(self, entity_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.is_available:
            return []
        try:
            cursor = await asyncio.wait_for(
                self._db.execute(
                    """SELECT r.id, r.source_id, r.target_id, r.relation_type, r.weight,
                              src.name AS source_name, tgt.name AS target_name
                       FROM relations r
                       JOIN entities src ON r.source_id = src.id
                       JOIN entities tgt ON r.target_id = tgt.id
                       WHERE r.source_id = ? OR r.target_id = ?
                       ORDER BY r.weight DESC LIMIT ?""",
                    (entity_id, entity_id, limit),
                ),
                timeout=CODABASE_QUERY_TIMEOUT,
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except asyncio.TimeoutError:
            logger.warning("CodebaseDBAdapter relations search timed out for entity: %s", entity_id)
            return []
        except Exception as e:
            logger.warning("CodebaseDBAdapter relations search failed: %s", e)
            return []
