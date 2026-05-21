import asyncio
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import chromadb

logger = logging.getLogger(__name__)


class ChromaMemoryBackend:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.memories = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )
        self.audit = self.client.get_or_create_collection(
            name="audit_logs",
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "user",
    ) -> str:
        memory_id = str(uuid.uuid4())
        meta = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            **(metadata or {}),
        }

        def _add():
            self.memories.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )

        await asyncio.to_thread(_add)
        await asyncio.to_thread(self._log_action, "add", memory_id, user_id, content)
        return memory_id

    async def _get(self, **kwargs):
        def _call():
            return self.memories.get(**kwargs)
        return await asyncio.to_thread(_call)

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        try:
            existing = await self._get(ids=[memory_id])
            if not existing["ids"]:
                logger.warning("Memory %s not found for metadata update", memory_id)
                return False
            existing_meta = existing["metadatas"][0] or {}
            existing_meta.update(metadata)

            def _update():
                self.memories.update(ids=[memory_id], metadatas=[existing_meta])

            await asyncio.to_thread(_update)
            logger.info("Metadata updated for memory %s: %s", memory_id, metadata)
            return True
        except Exception as e:
            logger.error("Metadata update failed for %s: %s", memory_id, e)
            return False

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
        level_filter: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if level_filter:
            where: Dict[str, Any] = {"$and": [{"user_id": user_id}, {"level": {"$in": level_filter}}]}
        else:
            where: Dict[str, Any] = {"user_id": user_id}

        def _query():
            return self.memories.query(
                query_embeddings=[embedding],
                where=where,
                n_results=top_k,
            )

        results = await asyncio.to_thread(_query)
        memories = []
        for i, mem_id in enumerate(results["ids"][0]):
            memories.append({
                "id": mem_id,
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return memories

    async def delete(self, memory_id: str) -> bool:
        try:
            def _delete():
                self.memories.delete(ids=[memory_id])

            await asyncio.to_thread(_delete)
            await asyncio.to_thread(self._log_action, "delete", memory_id, "", "")
            return True
        except Exception as e:
            logger.error("Delete failed: %s", e)
            return False

    async def update(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        try:
            existing = await self._get(ids=[memory_id])
            if not existing["ids"]:
                logger.warning("Memory %s not found for update", memory_id)
                return False
            existing_meta = existing["metadatas"][0] or {}
            existing_meta.update(metadata)

            def _update():
                self.memories.update(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[existing_meta],
                )

            await asyncio.to_thread(_update)
            logger.info("Memory updated: %s", memory_id)
            return True
        except Exception as e:
            logger.error("Update failed for %s: %s", memory_id, e)
            return False

    async def get_with_embeddings(
        self, user_id: str, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        results = await self._get(
            where={"user_id": user_id},
            limit=limit,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        return [
            {
                "id": results["ids"][i],
                "content": results["documents"][i],
                "embedding": results["embeddings"][i],
                "metadata": results["metadatas"][i],
            }
            for i in range(len(results["ids"]))
        ]

    async def get_with_embeddings_by_ids(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        results = await self._get(
            ids=ids,
            include=["embeddings", "documents", "metadatas"],
        )
        return [
            {
                "id": results["ids"][i],
                "user_id": results["metadatas"][i].get("user_id", "") if results["metadatas"][i] else "",
                "content": results["documents"][i],
                "embedding": results["embeddings"][i],
                "metadata": results["metadatas"][i],
                "level": results["metadatas"][i].get("level", "user") if results["metadatas"][i] else "user",
            }
            for i in range(len(results["ids"]))
        ]

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        all_data = await self._get(
            where={"user_id": user_id},
            limit=limit,
            offset=offset,
        )
        memories = []
        for i, mem_id in enumerate(all_data["ids"]):
            memories.append({
                "id": mem_id,
                "content": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            })
        return memories

    def _log_action(self, action: str, memory_id: str, user_id: str, content: str):
        log_entry = {
            "action": action,
            "memory_id": memory_id,
            "user_id": user_id,
            "content": content[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.audit.add(
            ids=[str(uuid.uuid4())],
            documents=[json.dumps(log_entry)],
            metadatas=[{"action": action, "user_id": user_id}],
        )
        self.audit = self.client.get_or_create_collection(
            name="audit_logs",
            metadata={"hnsw:space": "cosine"}
        )

