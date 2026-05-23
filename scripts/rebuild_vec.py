"""Rebuild vector and FTS indexes from existing memories.

Run after manually modifying the memories table (e.g. DELETE operations)
to ensure vec_memories and memory_fts are fully consistent.
"""
import asyncio
import json
import sys
from pathlib import Path

import aiosqlite
import sqlite_vec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.memorymesh.config import AppConfig
from src.memorymesh.embedder import get_embedding, prewarm_embedder
from src.memorymesh.memory.sqlite_vec_backend import normalize_l2


async def connect_db(db_path: Path) -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    conn = db._conn
    await db._execute(
        lambda: (
            conn.enable_load_extension(True),
            sqlite_vec.load(conn),
            conn.enable_load_extension(False),
        )
    )
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA cache_size = -20000")
    return db


async def ensure_vec_table(db: aiosqlite.Connection, dim: int):
    cursor = await db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'"
    )
    row = await cursor.fetchone()
    if row:
        expected = f"FLOAT[{dim}]"
        if expected not in row[0]:
            await db.execute("DROP TABLE IF EXISTS vec_memories")
            await db.execute(f"""
                CREATE VIRTUAL TABLE vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dim}]
                )
            """)
    else:
        await db.execute(f"""
            CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
        """)
    await db.commit()


async def rebuild():
    config = AppConfig.from_env()
    db_path = Path(config.sqlite_vec.db_path)
    if not db_path.exists():
        print(f"[fail] Database not found at {db_path}")
        sys.exit(1)

    print(f"[ .. ] Prewarming embedder ({config.embedding_model})...")
    await prewarm_embedder(config.embedding_model)

    print(f"[ .. ] Connecting to {db_path}...")
    db = await connect_db(db_path)

    # 1. Read all non-deleted memories
    cursor = await db.execute(
        """SELECT id, user_id, content, metadata_json, level, importance, created_at
           FROM memories WHERE deleted = 0 ORDER BY created_at ASC"""
    )
    rows = await cursor.fetchall()
    print(f"[ ok ] Found {len(rows)} memories to re-index")

    if not rows:
        print("[skip] Nothing to rebuild")
        await db.close()
        return

    # 2. Drop and recreate vec_memories + memory_fts
    print("[ .. ] Dropping old vec_memories and memory_fts...")
    await db.execute("DROP TABLE IF EXISTS vec_memories")
    await db.execute("DROP TABLE IF EXISTS memory_fts")
    await db.execute("DROP TRIGGER IF EXISTS trg_archive_cleanup")

    await db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            memory_id UNINDEXED, user_id UNINDEXED, level UNINDEXED,
            content, tokenize='unicode61'
        )
    """)
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_archive_cleanup
        AFTER UPDATE OF deleted ON memories
        WHEN NEW.deleted = 1
        BEGIN
            DELETE FROM vec_memories WHERE memory_id = OLD.id;
            DELETE FROM memory_fts   WHERE memory_id = OLD.id;
        END
    """)
    await db.commit()

    # 3. Re-index each memory
    inserted = 0
    first_embedding = None
    for row in rows:
        memory_id = row["id"]
        user_id = row["user_id"]
        content = row["content"]
        level = row["level"]

        try:
            embedding = await get_embedding(content, config.embedding_model)
        except Exception as e:
            print(f"[warn] Embedding failed for {memory_id}: {e}")
            continue

        if first_embedding is None:
            first_embedding = embedding
            await ensure_vec_table(db, len(embedding))

        normalized = normalize_l2(embedding)
        vec_bytes = np.array(normalized, dtype=np.float32).tobytes()

        await db.execute(
            "INSERT OR IGNORE INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (memory_id, vec_bytes),
        )
        await db.execute(
            "INSERT OR IGNORE INTO memory_fts(memory_id, user_id, level, content) VALUES (?, ?, ?, ?)",
            (memory_id, user_id, level, content),
        )
        inserted += 1

        if inserted % 20 == 0:
            await db.commit()
            print(f"[ .. ] {inserted}/{len(rows)} re-indexed...")

    await db.commit()
    await db.close()
    print(f"[ ok ] Re-indexed {inserted}/{len(rows)} memories successfully!")


if __name__ == "__main__":
    asyncio.run(rebuild())
