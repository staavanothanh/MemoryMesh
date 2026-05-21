import asyncio
import logging
from typing import List, Dict, Any, Optional

from ..config import AppConfig
from .chroma_impl import ChromaMemoryBackend
from .fts_backend import FTSBackend
from .hybrid_utils import RRFWithWeights

logger = logging.getLogger(__name__)


class HybridBackend:
    def __init__(self, config: AppConfig):
        self.chroma = ChromaMemoryBackend(config.chroma.db_path)
        self.fts = FTSBackend(config.fts.db_path)
        self.fuser = RRFWithWeights(
            k=config.rrf_k,
            weight_vec=config.rrf_weight_vec,
            weight_fts=config.rrf_weight_fts,
        )
        self._pool_size = config.rrf_pool_size

    async def initialize(self):
        await self.fts.initialize()

    async def close(self):
        await self.fts.close()

    async def add(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "user",
    ) -> str:
        """Gated write: chroma first, FTS second, rollback chroma if FTS fails."""
        memory_id = await self.chroma.add(user_id, content, embedding, metadata, level=level)
        try:
            await self.fts.add(memory_id, content, user_id, level=level)
        except Exception as e:
            logger.error("FTS add failed for %s, rolling back chroma entry: %s", memory_id, e)
            try:
                await self.chroma.delete(memory_id)
            except Exception as rollback_err:
                logger.error("Rollback failed for %s — orphaned in chroma: %s", memory_id, rollback_err)
            raise RuntimeError(f"Hybrid write failed: FTS error after chroma write: {e}") from e
        return memory_id

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        pool = max(self._pool_size, top_k * 2)

        chroma_task = asyncio.create_task(
            self.chroma.search(embedding, user_id, pool, level_filter=level_filter)
        )

        if query_text:
            try:
                fts_results = await self.fts.search(query_text, user_id, limit=pool, level_filter=level_filter)
            except Exception as e:
                logger.warning("FTS search failed, falling back to vector-only: %s", e)
                fts_results = []
        else:
            fts_results = []

        vector_results = await chroma_task

        if not fts_results:
            return vector_results[:top_k]

        return self.fuser.fuse(vector_results, fts_results, top_k=top_k)

    async def delete(self, memory_id: str) -> bool:
        """Gated delete: save backup data before chroma delete, rollback if FTS fails."""
        backup = None
        try:
            results = await self.chroma.get_with_embeddings_by_ids([memory_id])
            if results:
                backup = results[0]
        except Exception:
            pass

        success = await self.chroma.delete(memory_id)
        if not success:
            return False

        try:
            await self.fts.delete(memory_id)
        except Exception as e:
            logger.error("FTS delete failed for %s, rolling back chroma delete: %s", memory_id, e)
            if backup:
                try:
                    await self.chroma.add(
                        user_id=backup.get("user_id", ""),
                        content=backup.get("content", ""),
                        embedding=backup.get("embedding", [0.0]),
                        metadata=backup.get("metadata"),
                        level=backup.get("level", "user"),
                    )
                except Exception as rollback_err:
                    logger.error("Rollback re-insert failed for %s: %s", memory_id, rollback_err)
            raise RuntimeError(f"Hybrid delete failed: FTS error after chroma delete: {e}") from e
        return True

    async def update(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Gated update: chroma first, FTS second, no rollback needed (chroma is source of truth)."""
        success = await self.chroma.update(memory_id, content, metadata)
        if success:
            try:
                await self.fts.update(memory_id, content)
            except Exception as e:
                logger.warning("FTS update failed for %s — chroma still updated: %s", memory_id, e)
        return success

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        return await self.chroma.update_metadata(memory_id, metadata)

    async def get_with_embeddings(
        self, user_id: str, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.chroma.get_with_embeddings(user_id, limit, offset)

    async def get_with_embeddings_by_ids(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        return await self.chroma.get_with_embeddings_by_ids(ids)

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.chroma.list_all(user_id, limit, offset)
