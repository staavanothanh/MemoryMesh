import json
import re
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

MAX_RELATIONS_PER_QUERY = 20


class GraphStore:
    def __init__(self, db: aiosqlite.Connection, write_lock: asyncio.Lock):
        self._db = db
        self._write_lock = write_lock

    async def create_schema(self):
        from .sqlite_vec_backend import TransactionContext
        async with self._write_lock:
            async with TransactionContext(self._db):
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id         TEXT PRIMARY KEY,
                        user_id    TEXT NOT NULL,
                        name       TEXT NOT NULL,
                        type       TEXT NOT NULL DEFAULT 'concept',
                        properties TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id)"
                )
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(user_id, name)"
                )
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS relations (
                        id            TEXT PRIMARY KEY,
                        user_id       TEXT NOT NULL,
                        source_id     TEXT NOT NULL,
                        target_id     TEXT NOT NULL,
                        relation_type TEXT NOT NULL,
                        weight        REAL NOT NULL DEFAULT 1.0,
                        metadata      TEXT NOT NULL DEFAULT '{}',
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL,
                        FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                        FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                    )
                """)
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_user ON relations(user_id)"
                )
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)"
                )
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)"
                )
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(user_id, relation_type)"
                )
                await self._db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(
                        entity_id UNINDEXED,
                        user_id   UNINDEXED,
                        name,
                        type      UNINDEXED
                    )
                """)

    # ── Entity CRUD ──────────────────────────────────────────────────────

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """Normalize entity name for dedup: lowercase, strip, collapse whitespace, normalize separators."""
        normalized = name.lower().strip()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = " ".join(normalized.split())
        return normalized

    async def create_entity(
        self,
        name: str,
        user_id: str,
        entity_type: str = "concept",
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        norm = self.normalize_entity_name(name)
        existing = await self.find_entity_by_normalized(norm, user_id)
        if existing:
            logger.info("Entity already exists (normalized match): %s -> %s", name, existing["name"])
            return existing["id"]
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        from .sqlite_vec_backend import TransactionContext
        async with self._write_lock:
            async with TransactionContext(self._db):
                await self._db.execute(
                    """INSERT INTO entities (id, user_id, name, type, properties, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (entity_id, user_id, name, entity_type, json.dumps(properties or {}), now, now),
                )
                await self._db.execute(
                    "INSERT INTO entity_fts (entity_id, user_id, name, type) VALUES (?, ?, ?, ?)",
                    (entity_id, user_id, name, entity_type),
                )
        logger.info("Entity created: %s (name=%s, type=%s)", entity_id[:12], name, entity_type)
        return entity_id

    async def find_entity_by_normalized(self, normalized_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM entities WHERE user_id = ?",
            (user_id,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            if self.normalize_entity_name(row["name"]) == normalized_name:
                return self._row_to_entity(row)
        return None

    async def get_entity_by_name(self, name: str, user_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM entities WHERE name = ? AND user_id = ? LIMIT 1",
            (name, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    async def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    async def list_entities(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM entities WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def search_entities_fts(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cleaned = " ".join(re.findall(r"[\w\s]+", query)).strip()
        if not cleaned:
            return []
        cursor = await self._db.execute(
            "SELECT entity_id, name, type, rank FROM entity_fts WHERE user_id = ? AND name MATCH ? ORDER BY rank ASC LIMIT ?",
            (user_id, cleaned, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            entity = await self.get_entity_by_id(row["entity_id"])
            if entity:
                results.append(entity)
        return results

    async def merge_entities(self, source_name: str, target_name: str, user_id: str) -> Dict[str, Any]:
        """Merge source entity into target entity. All relations from source are re-pointed to target."""
        source = await self.get_entity_by_name(source_name, user_id)
        target = await self.get_entity_by_name(target_name, user_id)
        if not source:
            return {"success": False, "error": f"Source entity '{source_name}' not found"}
        if not target:
            return {"success": False, "error": f"Target entity '{target_name}' not found"}
        if source["id"] == target["id"]:
            return {"success": False, "error": "Source and target entities are the same"}

        now = datetime.now(timezone.utc).isoformat()
        source_id = source["id"]
        target_id = target["id"]

        from .sqlite_vec_backend import TransactionContext
        async with self._write_lock:
            async with TransactionContext(self._db):
                cursor = await self._db.execute(
                    "UPDATE relations SET source_id = ?, updated_at = ? WHERE source_id = ?",
                    (target_id, now, source_id),
                )
                out_count = cursor.rowcount
                cursor = await self._db.execute(
                    "UPDATE relations SET target_id = ?, updated_at = ? WHERE target_id = ?",
                    (target_id, now, source_id),
                )
                in_count = cursor.rowcount
                await self._db.execute("DELETE FROM entity_fts WHERE entity_id = ?", (source_id,))
                await self._db.execute(
                    "DELETE FROM entities WHERE id = ?", (source_id,),
                )

        logger.info(
            "Merged entity '%s' -> '%s': %d outgoing, %d incoming relations migrated",
            source_name, target_name, out_count, in_count,
        )
        return {
            "success": True,
            "target": target_name,
            "source": source_name,
            "relations_migrated": out_count + in_count,
        }

    async def delete_entity(self, entity_id: str, user_id: str) -> bool:
        from .sqlite_vec_backend import TransactionContext
        async with self._write_lock:
            async with TransactionContext(self._db):
                cursor = await self._db.execute(
                    "DELETE FROM entities WHERE id = ? AND user_id = ?",
                    (entity_id, user_id),
                )
                await self._db.execute(
                    "DELETE FROM entity_fts WHERE entity_id = ?",
                    (entity_id,),
                )
                deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Entity deleted: %s", entity_id[:12])
        return deleted

    # ── Relation CRUD ────────────────────────────────────────────────────

    async def create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        user_id: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        relation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        from .sqlite_vec_backend import TransactionContext
        async with self._write_lock:
            async with TransactionContext(self._db):
                await self._db.execute(
                    """INSERT INTO relations (id, user_id, source_id, target_id, relation_type, weight, metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (relation_id, user_id, source_id, target_id, relation_type, weight,
                     json.dumps(metadata or {}), now, now),
                )
        logger.info("Relation created: %s (%s -> %s)", relation_id[:12], source_id[:12], target_id[:12])
        return relation_id

    async def get_relations_for_entity(
        self, entity_id: str, user_id: str, limit: int = MAX_RELATIONS_PER_QUERY
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT r.*, src.name as source_name, tgt.name as target_name
               FROM relations r
               JOIN entities src ON r.source_id = src.id
               JOIN entities tgt ON r.target_id = tgt.id
               WHERE (r.source_id = ? OR r.target_id = ?) AND r.user_id = ?
               ORDER BY r.weight DESC, r.updated_at DESC
               LIMIT ?""",
            (entity_id, entity_id, user_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_relation(r) for r in rows]

    # ── Graph Traversal: Recursive CTE ───────────────────────────────────

    async def trace_path(
        self,
        source_name: str,
        target_name: str,
        user_id: str,
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        src_entity = await self.get_entity_by_name(source_name, user_id)
        if not src_entity:
            return []
        tgt_entity = await self.get_entity_by_name(target_name, user_id)
        if not tgt_entity:
            return []

        cursor = await self._db.execute(
            """WITH RECURSIVE path_cte(source_id, target_id, relation_type, depth, path_ids) AS (
                SELECT r.source_id, r.target_id, r.relation_type, 1,
                       json_array(r.source_id, r.target_id)
                FROM relations r
                WHERE r.source_id = ? AND r.user_id = ?
                UNION ALL
                SELECT r.source_id, r.target_id, r.relation_type, p.depth + 1,
                       json_set(p.path_ids, '$[#]', r.target_id)
                FROM relations r
                JOIN path_cte p ON r.source_id = p.target_id
                WHERE p.depth < ?
                  AND json_each_type(p.path_ids, '$') IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM json_each(p.path_ids) WHERE value = r.target_id
                  )
            )
            SELECT pcte.*, src.name AS source_name, tgt.name AS target_name
            FROM path_cte pcte
            JOIN entities src ON pcte.source_id = src.id
            JOIN entities tgt ON pcte.target_id = tgt.id
            WHERE pcte.target_id = ?
            ORDER BY pcte.depth ASC
            LIMIT 10""",
            (src_entity["id"], user_id, max_depth, tgt_entity["id"]),
        )
        rows = await cursor.fetchall()
        return [self._row_to_path_step(r) for r in rows]

    async def trace_entity(
        self,
        entity_name: str,
        user_id: str,
        max_depth: int = 3,
        max_relations: int = MAX_RELATIONS_PER_QUERY,
    ) -> Dict[str, Any]:
        entity = await self.get_entity_by_name(entity_name, user_id)
        if not entity:
            return {"entity": None, "relations": [], "error": f"Entity '{entity_name}' not found"}

        seen_relations = set()
        seen_nodes = {entity["id"]}
        all_relations: List[Dict] = []
        current_ids = [entity["id"]]

        for depth in range(max_depth):
            if not current_ids or len(all_relations) >= max_relations:
                break
            placeholders = ",".join("?" for _ in current_ids)
            cursor = await self._db.execute(
                f"""SELECT r.*, src.name as source_name, tgt.name as target_name
                    FROM relations r
                    JOIN entities src ON r.source_id = src.id
                    JOIN entities tgt ON r.target_id = tgt.id
                    WHERE (r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders}))
                      AND r.user_id = ?
                    ORDER BY r.weight DESC, r.updated_at DESC""",
                current_ids + current_ids + [user_id],
            )
            rows = await cursor.fetchall()
            next_ids = []
            for row in rows:
                rel = self._row_to_relation(row)
                rel_key = rel["id"]
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    all_relations.append(rel)
                    neighbor = rel["target_id"] if rel["source_id"] in current_ids else rel["source_id"]
                    if neighbor not in seen_nodes:
                        seen_nodes.add(neighbor)
                        next_ids.append(neighbor)
                    if len(all_relations) >= max_relations:
                        break
            current_ids = list(set(next_ids))

        return {
            "entity": entity,
            "relations": all_relations[:max_relations],
        }

    # ── Query graph: 1-hop neighbors ─────────────────────────────────────

    async def query_graph(
        self, entity_name: str, user_id: str, limit: int = MAX_RELATIONS_PER_QUERY
    ) -> Dict[str, Any]:
        entity = await self.get_entity_by_name(entity_name, user_id)
        if not entity:
            return {"entity": None, "neighbors": [], "error": f"Entity '{entity_name}' not found"}

        relations = await self.get_relations_for_entity(entity["id"], user_id, limit)
        neighbors = []
        for rel in relations:
            is_source = rel["source_id"] == entity["id"]
            neighbor_name = rel["target_name"] if is_source else rel["source_name"]
            neighbors.append({
                "entity": neighbor_name,
                "relation": rel["relation_type"],
                "direction": "outgoing" if is_source else "incoming",
                "weight": rel["weight"],
            })
        return {
            "entity": entity,
            "neighbors": neighbors,
        }

    # ── Hybrid search: join graph with memory tags ───────────────────────

    async def search_by_entity_tag(
        self, entity_name: str, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        cursor = await self._db.execute(
            """SELECT m.id, m.content, m.metadata_json
               FROM memories m
               WHERE m.user_id = ? AND m.deleted = 0
                 AND json_extract(m.metadata_json, '$.tags') LIKE ?
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (user_id, f'%"{entity_name.lower()}"%', limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    # ── Format helpers ───────────────────────────────────────────────────

    @staticmethod
    def format_xml_triplet(relations: List[Dict[str, Any]]) -> str:
        if not relations:
            return ""
        parts = ["<KnowledgeGraph>"]
        for rel in relations:
            parts.append(
                f'  <Relation source="{rel.get("source_name", "?")}" '
                f'type="{rel.get("relation_type", "?")}" '
                f'target="{rel.get("target_name", "?")}" />'
            )
        parts.append("</KnowledgeGraph>")
        return "\n".join(parts)

    @staticmethod
    def format_xml_from_neighbors(
        entity: Dict[str, Any], neighbors: List[Dict[str, Any]]
    ) -> str:
        if not neighbors:
            return ""
        parts = [f"<KnowledgeGraph root=\"{entity['name']}\">"]
        for nb in neighbors:
            if nb["direction"] == "outgoing":
                parts.append(
                    f'  <Relation source="{entity["name"]}" '
                    f'type="{nb["relation"]}" '
                    f'target="{nb["entity"]}" />'
                )
            else:
                parts.append(
                    f'  <Relation source="{nb["entity"]}" '
                    f'type="{nb["relation"]}" '
                    f'target="{entity["name"]}" />'
                )
        parts.append("</KnowledgeGraph>")
        return "\n".join(parts)

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_entity(row: aiosqlite.Row) -> Dict[str, Any]:
        d = dict(row)
        d["properties"] = json.loads(d.get("properties", "{}"))
        return d

    @staticmethod
    def _row_to_relation(row: aiosqlite.Row) -> Dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return d

    @staticmethod
    def _row_to_path_step(row: aiosqlite.Row) -> Dict[str, Any]:
        return {
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "target_id": row["target_id"],
            "target_name": row["target_name"],
            "relation_type": row["relation_type"],
            "depth": row["depth"],
        }
