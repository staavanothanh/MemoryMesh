import asyncio
import logging
from typing import List, Optional

import tiktoken

from ..config import AppConfig
from ..errors import ValidationError
from ..embedder import get_embedding
from ..router import RouterClient
from ..schemas import MemoryRecord, SearchResult
from .backend import MemoryBackend
from ..prompts import EXTRACT_METADATA_PROMPT

logger = logging.getLogger(__name__)

MAX_RECALL_TOKENS = 1000  # Token budget for recall results

class MemoryManager:
    def __init__(self, config: AppConfig, backend: MemoryBackend, router: RouterClient):
        self.config = config
        self.backend = backend
        self.router = router
        self._write_lock = asyncio.Lock()
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # phù hợp DeepSeek

    async def add_memory(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        importance: int = 3,
        user_id: str = None,
    ) -> str:
        """Add a memory with fast path and background enrichment."""
        user_id = user_id or self.config.default_user_id
        # Validate
        if len(text) > self.config.max_memory_length:
            raise ValidationError(f"Memory content exceeds max length {self.config.max_memory_length}")
        if importance < 1 or importance > 5:
            raise ValidationError("Importance must be between 1 and 5")

        # Compute embedding
        embedding = await get_embedding(text, self.config.embedding_model)

        # Fast path: store immediately
        async with self._write_lock:
            memory_id = await self.backend.add(
                user_id=user_id,
                content=text,
                embedding=embedding,
                metadata={
                    "tags": tags or [],
                    "importance": importance,
                },
            )
        logger.info("Memory saved: %s", memory_id)

        # Background enrichment (non-blocking)
        asyncio.create_task(self._enrich_memory(memory_id, text, user_id))
        return memory_id

    async def _enrich_memory(self, memory_id: str, text: str, user_id: str):
        """Call LLM to get tags, importance, summary and update memory metadata."""
        try:
            prompt = EXTRACT_METADATA_PROMPT.format(content=text)
            response = await self.router.call_llm(prompt)
            import json
            meta = json.loads(response)
            # Cập nhật metadata trong ChromaDB (sẽ cần phương thức update)
            # Hiện tại ChromaDB chưa có update metadata dễ dàng, ta tạm lưu log
            logger.info("Enrichment for %s: %s", memory_id, meta)
            # Trong phiên bản đầy đủ sẽ gọi self.backend.update_metadata(memory_id, meta)
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", memory_id, e)

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = None,
    ) -> List[SearchResult]:
        """Recall memories with token budget."""
        user_id = user_id or self.config.default_user_id
        embedding = await get_embedding(query, self.config.embedding_model)
        results = await self.backend.search(embedding, user_id, top_k)

        # Token budget: cắt bớt nội dung nếu cần
        limited_results = []
        total_tokens = 0
        for mem in results:
            tokens = len(self._tokenizer.encode(mem["content"]))
            if total_tokens + tokens > MAX_RECALL_TOKENS:
                # Cắt nội dung để vừa token budget
                available = MAX_RECALL_TOKENS - total_tokens
                truncated = self._tokenizer.decode(self._tokenizer.encode(mem["content"])[:available])
                mem["content"] = truncated + "..."
                limited_results.append(mem)
                break
            limited_results.append(mem)
            total_tokens += tokens

        return [
            SearchResult(
                id=m["id"],
                content=m["content"],
                score=m["score"],
                tags=m["metadata"].get("tags", []),
                importance=m["metadata"].get("importance", 3),
                timestamp=m["metadata"].get("timestamp", ""),
            )
            for m in limited_results
        ]

    async def forget_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        async with self._write_lock:
            success = await self.backend.delete(memory_id)
        if success:
            logger.info("Memory forgotten: %s", memory_id)
        return success

    async def list_memories(
        self, limit: int = 100, offset: int = 0, user_id: str = None
    ) -> List[MemoryRecord]:
        """List memories for a user."""
        user_id = user_id or self.config.default_user_id
        results = await self.backend.list_all(user_id, limit, offset)
        return [
            MemoryRecord(
                id=m["id"],
                user_id=user_id,
                content=m["content"],
                tags=m["metadata"].get("tags", []),
                importance=m["metadata"].get("importance", 3),
                timestamp=m["metadata"].get("timestamp", ""),
            )
            for m in results
        ]