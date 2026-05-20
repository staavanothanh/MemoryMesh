import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

import tiktoken

from ..config import AppConfig
from ..errors import ValidationError
from ..embedder import get_embedding
from ..router import RouterClient
from ..schemas import MemoryRecord, SearchResult
from ..hooks import HookRegistry
from .backend import MemoryBackend
from .consolidation import ConsolidationEngine
from ..prompts import EXTRACT_METADATA_PROMPT

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(
        self,
        config: AppConfig,
        backend: MemoryBackend,
        router: RouterClient,
        hooks: Optional[HookRegistry] = None,
    ):
        self.config = config
        self.backend = backend
        self.router = router
        self.hooks = hooks
        self._write_lock = asyncio.Lock()
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # phù hợp DeepSeek
        self._consolidator = ConsolidationEngine(config, backend, router)

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
            metadata = {"importance": importance}
            if tags:
                metadata["tags"] = tags
            memory_id = await self.backend.add(
                user_id=user_id,
                content=text,
                embedding=embedding,
                metadata=metadata,
            )
        logger.info("Memory saved: %s", memory_id)

        # Background enrichment (non-blocking)
        asyncio.create_task(self._enrich_memory(memory_id, text, user_id))

        # Background consolidation check (non-blocking)
        asyncio.create_task(self._maybe_consolidate(user_id))

        # Trigger post-tool hooks
        if self.hooks:
            asyncio.create_task(
                self.hooks.trigger("after_remember", memory_id=memory_id, user_id=user_id)
            )

        return memory_id

    async def _enrich_memory(self, memory_id: str, text: str, user_id: str):
        """Call LLM to get tags, importance, summary and update memory metadata."""
        try:
            prompt = EXTRACT_METADATA_PROMPT.format(content=text)
            response = await self.router.call_llm(prompt)
            import json
            meta = json.loads(response)
            update = {}
            if "tags" in meta and isinstance(meta["tags"], list):
                update["tags"] = meta["tags"]
            if "importance" in meta and isinstance(meta["importance"], int):
                update["importance"] = meta["importance"]
            if "summary" in meta and isinstance(meta["summary"], str):
                update["summary"] = meta["summary"]
            if update:
                await self.backend.update_metadata(memory_id, update)
                logger.info("Enrichment updated for %s: %s", memory_id, update)
            else:
                logger.info("Enrichment produced no usable fields for %s: %s", memory_id, meta)
        except Exception as e:
            logger.error("Enrichment failed for %s: %s", memory_id, e)

    async def _maybe_consolidate(self, user_id: str):
        """Run consolidation if enabled, non-blocking."""
        try:
            merged = await self._consolidator.run_for_user(user_id)
            if merged:
                logger.info("Consolidation merged %d clusters for user %s", merged, user_id)
        except Exception as e:
            logger.error("Consolidation failed for user %s: %s", user_id, e)

    @staticmethod
    def _recency_score(timestamp_str: str) -> float:
        """Compute recency score (0-1) decaying over ~1 day."""
        try:
            created = datetime.fromisoformat(timestamp_str)
            hours_ago = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            return __import__("math").exp(-abs(hours_ago) / 24)
        except (ValueError, TypeError, OverflowError):
            return 0.0

    def _compute_final_score(self, mem: dict) -> float:
        w_s = self.config.truncation_weight_score
        w_i = self.config.truncation_weight_importance
        w_r = self.config.truncation_weight_recency

        fusion_score = mem.get("score", 0.0)
        importance = mem.get("metadata", {}).get("importance", 3) / 5.0
        recency = self._recency_score(mem.get("metadata", {}).get("timestamp", ""))
        return fusion_score * w_s + importance * w_i + recency * w_r

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = None,
    ) -> List[SearchResult]:
        """Recall with smart truncation: score = fusion * 0.6 + importance * 0.3 + recency * 0.1."""
        user_id = user_id or self.config.default_user_id
        embedding = await get_embedding(query, self.config.embedding_model)
        results = await self.backend.search(embedding, user_id, top_k, query_text=query)

        budget = self.config.token_budget

        # Smart re-ranking
        scored = [(self._compute_final_score(m), m) for m in results]
        scored.sort(key=lambda x: x[0], reverse=True)

        limited_results = []
        total_tokens = 0
        for _, mem in scored:
            meta_str = str(mem.get("metadata", {}))
            meta_tokens = len(self._tokenizer.encode(meta_str))
            content_tokens = len(self._tokenizer.encode(mem["content"]))
            mem_tokens = meta_tokens + content_tokens
            if total_tokens + mem_tokens > budget:
                available = budget - total_tokens
                if available > 10:
                    truncated = self._tokenizer.decode(self._tokenizer.encode(mem["content"])[:available])
                    mem["content"] = truncated + "..."
                    limited_results.append(mem)
                break
            limited_results.append(mem)
            total_tokens += mem_tokens

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