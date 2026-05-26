"""
Single-DB backend: vector (sqlite-vec) + FTS5 + metadata in one SQLite file.
Unified backend — vector (sqlite-vec), FTS5, and metadata in a single SQLite database.
"""
import json
import re
import math
import uuid
import random
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import aiosqlite
import sqlite_vec
import numpy as np

from .graph_store import GraphStore

logger = logging.getLogger(__name__)

DIM = 384


class TransactionContext:
    """Async context manager for safe SQLite transactions.

    Wraps BEGIN/COMMIT/ROLLBACK with proper exception handling.
    Retries on 'database is locked' with exponential backoff + jitter
    to support concurrent multi-agent access.
    Works with aiosqlite < 0.23 (no built-in transaction() method).
    """

    MAX_RETRIES = 5
    BASE_DELAY = 0.05  # 50ms

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def __aenter__(self):
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                await self._db.execute("BEGIN")
                return self
            except Exception as e:
                err_str = str(e).lower()
                if "database is locked" in err_str or "database is busy" in err_str:
                    last_error = e
                    delay = self.BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_error or RuntimeError("Transaction BEGIN failed after retries")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                await self._db.rollback()
            except Exception:
                pass
        else:
            await self._db.commit()


def sanitize_fts_query(query: str) -> str:
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    return " ".join(cleaned.split())


def normalize_l2(vector: List[float]) -> List[float]:
    squared_sum = sum(x ** 2 for x in vector)
    magnitude = math.sqrt(squared_sum)
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def load_vec_extension(conn):
    """Load sqlite-vec extension into a raw sqlite3 connection."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


class SqliteVecBackend:
    """Unified storage: vector ANN via sqlite-vec, keyword via FTS5, metadata in SQLite.

    All write operations are wrapped in a single ACID transaction.
    Uses subquery pattern for vec0 to avoid JOIN-WHERE-MATCH incompatibility.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self.graph: Optional[GraphStore] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("SELECT 1")
        try:
            conn = self._db._conn
        except AttributeError:
            raise RuntimeError(
                "aiosqlite internal API 'Connection._conn' not found. "
                "Pin dependency: pip install 'aiosqlite>=0.20,<0.23'"
            )
        await self._db._execute(
            lambda: (
                conn.enable_load_extension(True),
                sqlite_vec.load(conn),
                conn.enable_load_extension(False),
            )
        )
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA cache_size = -20000")
        await self._db.execute("PRAGMA mmap_size = 268435456")
        await self._db.execute("PRAGMA temp_store = MEMORY")
        self.graph = GraphStore(self._db)
        await self._create_schema()
        await self.graph.create_schema()
        await self._db.commit()
        logger.info("SqliteVecBackend initialized at %s", self.db_path)

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def _create_schema(self):
        # Table 1: metadata master
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                content       TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                level         TEXT NOT NULL DEFAULT 'user',
                importance    INTEGER DEFAULT 3,
                deleted       INTEGER DEFAULT 0,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_level ON memories(user_id, level)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_time ON memories(created_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_deleted ON memories(user_id, deleted)"
        )
        # Phase 6: Composite indexes for dynamic scoring with keyset pagination
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_score ON memories(importance DESC, created_at DESC, id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user_score ON memories(user_id, importance DESC, created_at DESC, id)"
        )
        # Table 2: vector ANN (sqlite-vec) — created lazily on first add() with correct DIM
        # Table 3: FTS5 full-text search
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                memory_id UNINDEXED,
                user_id   UNINDEXED,
                level     UNINDEXED,
                content,
                tokenize='unicode61'
            )
        """)
        # Table 4: audit trail
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                action          TEXT NOT NULL,
                memory_id       TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                content_preview TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            )
        """)
        # Trigger: auto-cleanup vec + fts on soft-delete
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_archive_cleanup
            AFTER UPDATE OF deleted ON memories
            WHEN NEW.deleted = 1
            BEGIN
                DELETE FROM vec_memories WHERE memory_id = OLD.id;
                DELETE FROM memory_fts   WHERE memory_id = OLD.id;
            END
        """)

    async def _ensure_vec_table(self, dim: int):
        """Lazily create vec0 table with the correct embedding dimension.

        If the table already exists with a different dimension, drop and
        recreate it to avoid dimension mismatch errors.
        """
        if self._db is None:
            return
        cursor = await self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        )
        row = await cursor.fetchone()
        if row:
            # Table exists — check if dimension matches
            existing_sql = row["sql"]
            expected = f"FLOAT[{dim}]"
            if expected not in existing_sql:
                logger.warning("vec_memories dimension mismatch: dropping and recreating (dim=%d)", dim)
                await self._db.execute("DROP TABLE IF EXISTS vec_memories")
                await self._db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                        memory_id TEXT PRIMARY KEY,
                        embedding FLOAT[{dim}]
                    )
                """)
        else:
            await self._db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dim}]
                )
            """)

    # ------------------------------------------------------------------
    # Core write — single ACID transaction
    # ------------------------------------------------------------------

    async def add(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "user",
    ) -> str:
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            "user_id": user_id,
            "level": level,
            "timestamp": now,
            **(metadata or {}),
        }
        importance = meta.get("importance", 3)
        normalized = normalize_l2(embedding)
        vec_bytes = np.array(normalized, dtype=np.float32).tobytes()

        await self._ensure_vec_table(len(embedding))

        async with TransactionContext(self._db):
            await self._db.execute(
                """INSERT INTO memories
                   (id, user_id, content, metadata_json, level, importance,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, user_id, content, json.dumps(meta), level, importance,
                 now, now),
            )
            await self._db.execute(
                "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                (memory_id, vec_bytes),
            )
            await self._db.execute(
                "INSERT INTO memory_fts(memory_id, user_id, level, content) VALUES (?, ?, ?, ?)",
                (memory_id, user_id, level, content),
            )
            await self._db.execute(
                """INSERT INTO audit_log(action, memory_id, user_id, content_preview, created_at)
                   VALUES ('add', ?, ?, ?, ?)""",
                (memory_id, user_id, content[:200], now),
            )
        return memory_id

    async def delete(self, memory_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        async with TransactionContext(self._db):
            cursor = await self._db.execute(
                "SELECT user_id FROM memories WHERE id = ?", (memory_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return False
            uid = row["user_id"]
            await self._db.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            await self._db.execute(
                "DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,)
            )
            await self._db.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,)
            )
            await self._db.execute(
                """INSERT INTO audit_log(action, memory_id, user_id, content_preview, created_at)
                   VALUES ('delete', ?, ?, '', ?)""",
                (memory_id, uid, now),
            )
        return True

    async def update(
        self, memory_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        existing = await self._get_by_id_full(memory_id)
        if not existing:
            return False
        merged_meta = {**existing["metadata"], **metadata, "updated_at": now}
        importance = metadata.get("importance", existing["importance"])
        async with TransactionContext(self._db):
            await self._db.execute(
                """UPDATE memories
                   SET content=?, metadata_json=?, importance=?, updated_at=?
                   WHERE id=?""",
                (content, json.dumps(merged_meta), importance, now, memory_id),
            )
            await self._db.execute(
                "UPDATE memory_fts SET content=? WHERE memory_id=?",
                (content, memory_id),
            )
        return True

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        existing = await self._get_by_id_full(memory_id)
        if not existing:
            return False
        merged = {**existing["metadata"], **metadata, "updated_at": now}
        importance = metadata.get("importance", existing["importance"])
        new_level = metadata.get("level", existing.get("level"))
        await self._db.execute(
            """UPDATE memories
               SET metadata_json=?, importance=?, level=?, updated_at=?
               WHERE id=?""",
            (json.dumps(merged), importance, new_level, now, memory_id),
        )
        await self._db.commit()
        return True

    async def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """Phase 7.3: Update the vector embedding for an existing memory (background)."""
        from .sqlite_vec_backend import normalize_l2
        normalized = normalize_l2(embedding)
        import numpy as np
        vec_bytes = np.array(normalized, dtype=np.float32).tobytes()
        try:
            await self._db.execute(
                "UPDATE vec_memories SET embedding = ? WHERE memory_id = ?",
                (vec_bytes, memory_id),
            )
            await self._db.commit()
            return True
        except Exception as e:
            logger.error("Embedding update failed for %s: %s", memory_id[:12], e)
            return False

    # ------------------------------------------------------------------
    # Metadata-only update (no content change — used by consolidate/expire)
    # ------------------------------------------------------------------

    async def _flag_memories(self, ids: List[str], flag: str, value: bool = True):
        """Batch-flag memories (consolidated, expired, archived, fact_resolved)."""
        if not ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in ids)
        # Read current metadata_json for each id
        cursor = await self._db.execute(
            f"SELECT id, metadata_json FROM memories WHERE id IN ({placeholders})",
            ids,
        )
        rows = await cursor.fetchall()
        async with TransactionContext(self._db):
            for row in rows:
                meta = json.loads(row["metadata_json"])
                meta[flag] = value
                meta["updated_at"] = now
                await self._db.execute(
                    "UPDATE memories SET metadata_json=?, updated_at=? WHERE id=?",
                    (json.dumps(meta), now, row["id"]),
                )

    async def soft_delete(self, memory_id: str) -> bool:
        """Mark memory as deleted (keeps vec + fts — trigger handles cleanup)."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            "UPDATE memories SET deleted=1, updated_at=? WHERE id=? AND deleted=0",
            (now, memory_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Search — ANN via subquery pattern
    # ------------------------------------------------------------------

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """ANN search with subquery pattern to avoid vec0 JOIN limitation.

        Step 1: pure ANN search against vec_memories (no filter)
        Step 2: filter by user_id / level / deleted in memories table
        """
        normalized = normalize_l2(embedding)
        vec_bytes = np.array(normalized, dtype=np.float32).tobytes()
        pool = max(top_k * 2, 10)

        await self._ensure_vec_table(len(embedding))

        cursor = await self._db.execute(
            "SELECT memory_id, distance FROM vec_memories WHERE embedding MATCH ? AND k = ?",
            (vec_bytes, pool),
        )
        vector_hits = await cursor.fetchall()
        if not vector_hits:
            return []

        id_distance = {row["memory_id"]: float(row["distance"]) for row in vector_hits}
        ids = list(id_distance.keys())
        placeholders = ",".join("?" for _ in ids)

        params: list = ids + [user_id]
        sql = f"""
            SELECT id, content, metadata_json, level, importance
            FROM memories
            WHERE id IN ({placeholders}) AND user_id = ? AND deleted = 0
        """
        if level_filter:
            lvl_ph = ",".join("?" for _ in level_filter)
            sql += f" AND level IN ({lvl_ph})"
            params.extend(level_filter)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            dist = id_distance.get(row["id"], 1.0)
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": max(0.0, 1.0 - dist),
                "metadata": json.loads(row["metadata_json"]),
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # FTS methods (formerly in FTSBackend)
    # ------------------------------------------------------------------

    async def fts_search(
        self,
        query_text: str,
        user_id: str,
        limit: int = 10,
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
                ORDER BY rank ASC LIMIT ?
            """
            params = (user_id, *level_filter, cleaned, limit)
        else:
            sql = """
                SELECT memory_id, content, rank
                FROM memory_fts
                WHERE user_id = ? AND content MATCH ?
                ORDER BY rank ASC LIMIT ?
            """
            params = (user_id, cleaned, limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {"id": row["memory_id"], "content": row["content"], "score": float(row["rank"])}
            for row in rows
        ]

    async def list_recent(
        self,
        user_id: str,
        limit: int = 10,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Chronological fallback — rowid DESC = insert order into FTS."""
        if level_filter:
            placeholders = ",".join("?" for _ in level_filter)
            sql = f"""
                SELECT memory_id, content
                FROM memory_fts
                WHERE user_id = ? AND level IN ({placeholders})
                ORDER BY rowid DESC LIMIT ?
            """
            params = (user_id, *level_filter, limit)
        else:
            sql = """
                SELECT memory_id, content
                FROM memory_fts
                WHERE user_id = ?
                ORDER BY rowid DESC LIMIT ?
            """
            params = (user_id, limit)

        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {"id": row["memory_id"], "content": row["content"], "score": 0.0}
            for row in rows
        ]

    # ------------------------------------------------------------------
    # List / enumerate
    # ------------------------------------------------------------------

    async def list_by_tag(self, user_id: str, tag: str) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT m.id, m.user_id, m.content, m.metadata_json, m.level, m.importance
               FROM memories m, json_each(json_extract(m.metadata_json, '$.tags'))
               WHERE m.user_id = ? AND json_each.value = ? AND m.deleted = 0""",
            (user_id, tag),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "content": r["content"],
                "metadata": json.loads(r["metadata_json"]),
                "level": r["level"],
                "importance": r["importance"],
            }
            for r in rows
        ]

    async def delete_by_tag(self, user_id: str, tag: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        async with TransactionContext(self._db):
            cursor = await self._db.execute(
                """SELECT m.id FROM memories m, json_each(json_extract(m.metadata_json, '$.tags'))
               WHERE m.user_id = ? AND json_each.value = ? AND m.deleted = 0
               ORDER BY m.rowid DESC""",
                (user_id, tag),
            )
            ids = [r[0] for r in await cursor.fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            await self._db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", ids
            )
            await self._db.execute(
                f"DELETE FROM vec_memories WHERE memory_id IN ({placeholders})", ids
            )
            await self._db.execute(
                f"DELETE FROM memory_fts WHERE memory_id IN ({placeholders})", ids
            )
        logger.info("Deleted %d memories by tag '%s' user '%s'", len(ids), tag, user_id)
        return len(ids)

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT id, content, metadata_json
               FROM memories
               WHERE user_id = ? AND deleted = 0
                 AND COALESCE(json_extract(metadata_json, '$.archived'), 0) = 0
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    async def list_all_with_deleted(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Like list_all but includes soft-deleted records (for admin / migration)."""
        cursor = await self._db.execute(
            """SELECT id, content, metadata_json, deleted
               FROM memories
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "deleted": bool(row["deleted"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Getters with embeddings (for consolidation / enrichment)
    # ------------------------------------------------------------------

    async def get_with_embeddings(
        self, user_id: str, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT m.id, m.content, m.metadata_json, m.level, v.embedding
               FROM memories m
               JOIN vec_memories v ON m.id = v.memory_id
               WHERE m.user_id = ? AND m.deleted = 0
               ORDER BY m.created_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "level": row["level"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32).tolist(),
            }
            for row in rows
        ]

    async def get_with_embeddings_by_ids(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cursor = await self._db.execute(
            f"""SELECT m.id, m.user_id, m.content, m.metadata_json, m.level, v.embedding
                FROM memories m
                JOIN vec_memories v ON m.id = v.memory_id
                WHERE m.id IN ({placeholders}) AND m.deleted = 0""",
            ids,
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "level": row["level"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32).tolist(),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT id, content, metadata_json FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
        }

    async def _get_by_id_full(self, memory_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT id, content, metadata_json, importance, level
               FROM memories WHERE id = ?""",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
            "importance": row["importance"],
            "level": row["level"],
        }
