"""
Migrate from ChromaDB + FTS5 to single sqlite-vec database.

Usage:
    python scripts/migrate_to_sqlite_vec.py [--chroma-path ./db/chroma] [--fts-path ./db/memory_fts.db] [--vec-path ./db/memory.db]

Requirements:
    pip install chromadb  (only needed for migration)
"""
import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate")

DIM = 0  # auto-detected from first valid embedding


def read_chroma(chroma_path: str) -> list[dict]:
    """Read all memories from a ChromaDB SQLite store directly."""
    db_path = os.path.join(chroma_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        logger.error("ChromaDB not found at %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Chroma stores collections in 'collections' table
    cur = conn.execute("SELECT id, name FROM collections")
    collections = {row["id"]: row["name"] for row in cur.fetchall()}
    logger.info("Found collections: %s", collections)

    # embedding_metadata: id TEXT, embedding_id TEXT, string_value TEXT, float_value REAL, int_value INTEGER
    # embedding_fulltext: id TEXT, embedding_id TEXT, string_value TEXT
    # embeddings: id TEXT, collection_id TEXT, embedding BLOB, updated_at TIMESTAMP
    # We need: id (UUID), embedding (BLOB), and metadata (user_id, content, etc.)

    cur = conn.execute("""
        SELECT e.id AS memory_id, e.embedding, em.string_value AS metadata_json
        FROM embeddings e
        LEFT JOIN embedding_metadata em ON e.id = em.id
        WHERE em.key = 'json'
    """)
    # Actually Chroma stores metadata in embedding_metadata with key strings
    # Let's use a different approach

    memories = []
    cur = conn.execute("""
        SELECT e.id AS chroma_id, e.embedding, ef.string_value AS document,
               em.string_value AS metadata_str
        FROM embeddings e
        LEFT JOIN embedding_fulltext ef ON ef.id = e.id
        LEFT JOIN embedding_metadata em ON em.id = e.id AND em.key = 'json'
    """)
    for row in cur.fetchall():
        try:
            meta = json.loads(row["metadata_str"]) if row["metadata_str"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        memories.append({
            "id": row["chroma_id"],
            "content": row["document"] or meta.get("content", ""),
            "metadata": meta,
            "embedding_blob": row["embedding"],
        })

    conn.close()
    logger.info("Read %d memories from ChromaDB", len(memories))
    return memories


def read_fts(fts_path: str) -> dict:
    """Read all rows from FTS5 content table."""
    if not os.path.isfile(fts_path):
        logger.error("FTS db not found at %s", fts_path)
        sys.exit(1)

    conn = sqlite3.connect(fts_path)
    conn.row_factory = sqlite3.Row

    # FTS5 content table: memory_id, user_id, level, content
    cur = conn.execute("SELECT memory_id, user_id, level, content FROM memory_fts")
    rows = cur.fetchall()
    conn.close()

    # Index by memory_id
    fts_map = {}
    for row in rows:
        fts_map[row["memory_id"]] = {
            "user_id": row["user_id"],
            "level": row["level"],
            "content": row["content"],
        }

    logger.info("Read %d rows from FTS5", len(fts_map))
    return fts_map


async def migrate(chroma_path: str, fts_path: str, vec_path: str):
    """Read old data, write to new sqlite-vec db."""
    from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend

    chroma_memories = read_chroma(chroma_path)
    fts_map = read_fts(fts_path)

    backend = SqliteVecBackend(vec_path)
    await backend.initialize()

    now = datetime.now(timezone.utc).isoformat()
    migrated = 0
    skipped = 0

    for mem in chroma_memories:
        mem_id = mem["id"]
        meta = mem["metadata"]
        content = mem["content"] or meta.get("content", "")
        embedding_blob = mem["embedding_blob"]

        if not content:
            skipped += 1
            continue

        # Decode embedding blob (Chroma stores float32 binary)
        import numpy as np
        try:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
        except Exception:
            skipped += 1
            continue

        # Auto-detect embedding dimension from first valid embedding
        if DIM == 0:
            DIM = len(embedding)
            logger.info("Auto-detected embedding dimension: %d", DIM)

        if len(embedding) != DIM:
            logger.warning("Skipping %s: embedding dim %d != %d", mem_id, len(embedding), DIM)
            skipped += 1
            continue

        # Get user_id and level from FTS map, or fallback to metadata
        fts_row = fts_map.get(mem_id, {})
        user_id = meta.get("user_id") or fts_row.get("user_id") or "Shinn"
        level = meta.get("level") or fts_row.get("level") or "user"

        # Insert directly via raw SQL to preserve original ID and timestamps
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
        meta_json = json.dumps(meta)
        importance = meta.get("importance", 3)
        created_at = meta.get("timestamp", now)
        updated_at = now

        async with backend._db.execute("BEGIN"):
            await backend._db.execute(
                """INSERT OR IGNORE INTO memories
                   (id, user_id, content, metadata_json, level, importance,
                    created_at, updated_at, deleted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (mem_id, user_id, content, meta_json, level, importance,
                 created_at, updated_at),
            )
            await backend._db.execute(
                "INSERT OR IGNORE INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                (mem_id, vec_bytes),
            )
            await backend._db.execute(
                "INSERT OR IGNORE INTO memory_fts(memory_id, user_id, level, content) VALUES (?, ?, ?, ?)",
                (mem_id, user_id, level, content),
            )
            await backend._db.execute(
                """INSERT INTO audit_log(action, memory_id, user_id, content_preview, created_at)
                   VALUES ('migrate', ?, ?, ?, ?)""",
                (mem_id, user_id, content[:200], now),
            )
        await backend._db.commit()
        migrated += 1

    await backend.close()
    logger.info("Migration complete: %d migrated, %d skipped", migrated, skipped)


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB + FTS to sqlite-vec")
    parser.add_argument("--chroma-path", default="./db/chroma")
    parser.add_argument("--fts-path", default="./db/memory_fts.db")
    parser.add_argument("--vec-path", default="./db/memory.db")
    args = parser.parse_args()

    asyncio.run(migrate(args.chroma_path, args.fts_path, args.vec_path))


if __name__ == "__main__":
    main()
