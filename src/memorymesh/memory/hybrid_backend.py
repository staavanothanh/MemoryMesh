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
        memory_id = await self.chroma.add(user_id, content, embedding, metadata, level=level)
        try:
            await self.fts.add(memory_id, content, user_id, level=level)
        except Exception as e:
            logger.warning("FTS index failed for %s, degraded to vector-only: %s", memory_id, e)
        return memory_id

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        pool = self._pool_size

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

        valid_ids = {r["id"] for r in vector_results}
        fts_filtered = [r for r in fts_results if r["id"] in valid_ids]

        if not fts_filtered:
            return vector_results[:top_k]

        return self.fuser.fuse(vector_results, fts_filtered, top_k=top_k)

    async def delete(self, memory_id: str) -> bool:
        success = await self.chroma.delete(memory_id)
        if success:
            await self.fts.delete(memory_id)
        return success

    async def update(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        success = await self.chroma.update(memory_id, content, metadata)
        if success:
            try:
                await self.fts.update(memory_id, content)
            except Exception as e:
                logger.warning("FTS update failed for %s: %s", memory_id, e)
        return success

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        return await self.chroma.update_metadata(memory_id, metadata)

    async def get_with_embeddings(
        self, user_id: str, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.chroma.get_with_embeddings(user_id, limit, offset)

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        return await self.chroma.list_all(user_id, limit, offset)
