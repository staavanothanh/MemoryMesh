import asyncio
import logging
import math
import time
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
from .instinct_store import InstinctStore
from .instinct import InstinctEngine
from ..prompts import EXTRACT_METADATA_PROMPT

_MAGENTA = "\033[1;35m"
_CYAN = "\033[1;36m"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"

logger = logging.getLogger(__name__)


def _log_bg(label: str, msg: str, emoji: str = ""):
    """ANSI-colored structured log for background operations."""
    logger.info("%s %s[%s]%s %s", emoji, _MAGENTA, label, _RESET, msg)

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
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._consolidator = ConsolidationEngine(config, backend, router)
        self.instinct_store = InstinctStore(config.instinct.db_path)
        self.instinct_engine = InstinctEngine(config, backend, self.instinct_store)
        self._last_consolidation: dict[str, float] = {}
        self._last_fact_resolution: dict[str, float] = {}
        self._last_expiry: dict[str, float] = {}
        self._last_fts_reconciliation: float = 0.0
        self._background_tasks: set[asyncio.Task] = set()

    def _create_tracked_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def shutdown(self):
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    @staticmethod
    def _is_workspace_visible(memory_path: Optional[str], current_path: str) -> bool:
        """Hierarchical workspace visibility.

        Rules:
        - Same path: visible
        - Siblings (same parent): visible
        - Current is ancestor of memory (downward): visible
        - Otherwise: not visible
        - Legacy memories (no path): visible from everywhere
        """
        if not memory_path:
            return True
        normalized_mem = memory_path.replace("\\", "/").rstrip("/")
        normalized_cur = current_path.replace("\\", "/").rstrip("/")
        if normalized_mem == normalized_cur:
            return True
        if normalized_mem.startswith(normalized_cur + "/"):
            return True
        import os
        if os.path.dirname(normalized_mem) == os.path.dirname(normalized_cur):
            return True
        return False

    async def add_memory(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        importance: int = 3,
        user_id: str = None,
        level: str = "user",
        workspace_path: Optional[str] = None,
    ) -> str:
        """Add a memory with fast path, instinct suggestions, and background enrichment."""
        user_id = user_id or self.config.default_user_id
        if level not in ("user", "session", "knowledge"):
            raise ValidationError("level must be one of: user, session, knowledge")
        if len(text) > self.config.max_memory_length:
            raise ValidationError(f"Memory content exceeds max length {self.config.max_memory_length}")
        if importance < 1 or importance > 5:
            raise ValidationError("Importance must be between 1 and 5")

        # Apply instinct suggestions for tags
        if tags is None:
            tags = []
        if self.config.instinct.enabled:
            try:
                suggestions = await self.instinct_engine.apply_instincts(user_id, text, tags)
                for s in suggestions.get("suggested_tags", []):
                    if s["tag"] not in tags and s["confidence"] > 0.5:
                        tags.append(s["tag"])
                        self._create_tracked_task(self.instinct_engine.reinforce_instinct(s["instinct_id"], success=True))
            except Exception as e:
                logger.warning("Instinct suggestion failed: %s", e)

        embedding = await get_embedding(text, self.config.embedding_model)

        async with self._write_lock:
            metadata = {"importance": importance, "level": level}
            if tags:
                metadata["tags"] = tags
            if workspace_path:
                metadata["workspace_path"] = workspace_path.replace("\\", "/").rstrip("/")
            memory_id = await self.backend.add(
                user_id=user_id,
                content=text,
                embedding=embedding,
                metadata=metadata,
                level=level,
            )
        logger.info("Memory saved: %s", memory_id)

        # Background tasks with rate-limiting
        self._create_tracked_task(self._run_background_tasks(memory_id, text, level, user_id))

        # Trigger post-tool hooks
        if self.hooks:
            self._create_tracked_task(
                self.hooks.trigger("after_remember", memory_id=memory_id, user_id=user_id)
            )

        return memory_id

    async def _run_background_tasks(self, memory_id: str, text: str, level: str, user_id: str):
        """Run background tasks with rate-limiting for expensive operations."""
        now = time.monotonic()

        # 1. Enrichment: skip for session-level (chat logs, auto-logs)
        if level != "session":
            self._create_tracked_task(self._enrich_memory(memory_id, text, user_id))

        # 2. Instinct learning: lightweight, no LLM
        if self.config.instinct.enabled:
            self._create_tracked_task(self._maybe_learn_instincts(user_id))

        # 3. Consolidation: rate-limited (max once per 60s per user)
        last_c = self._last_consolidation.get(user_id, 0)
        if now - last_c >= 60:
            self._last_consolidation[user_id] = now
            self._create_tracked_task(self._maybe_consolidate(user_id))

        # 4. Fact resolution: rate-limited (max once per 120s per user)
        last_f = self._last_fact_resolution.get(user_id, 0)
        if now - last_f >= 120:
            self._last_fact_resolution[user_id] = now
            self._create_tracked_task(self._maybe_resolve_facts(user_id))

        # 5. Session memory expiry: rate-limited (max once per 60s per user)
        last_e = self._last_expiry.get(user_id, 0)
        if now - last_e >= 60:
            self._last_expiry[user_id] = now
            self._create_tracked_task(self._maybe_expire_memories(user_id))

        # 6. FTS reconciliation: rate-limited (max once per 300s regardless of user)
        self._create_tracked_task(self._maybe_reconcile_fts())

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

    async def _maybe_expire_memories(self, user_id: str):
        """Expire old session memories, non-blocking."""
        try:
            expired = await self._consolidator.run_expiry(user_id)
            if expired:
                _log_bg("Expiry", f"Expired {expired} old session memories for {user_id}", emoji="")
        except Exception as e:
            logger.warning("Memory expiry failed for user %s: %s", user_id, e)

    async def _maybe_consolidate(self, user_id: str):
        """Run consolidation if enabled, non-blocking."""
        try:
            merged = await self._consolidator.run_for_user(user_id)
            if merged:
                _log_bg("Consolidation", f"Merged {merged} clusters for {user_id}", emoji="")
        except Exception as e:
            logger.error("Consolidation failed for user %s: %s", user_id, e)

    async def _maybe_resolve_facts(self, user_id: str):
        """Run fact contradiction resolution if enabled, non-blocking."""
        try:
            resolved = await self._consolidator.run_fact_consolidation(user_id)
            if resolved:
                _log_bg("FactResolve", f"Resolved {resolved} contradiction groups for {user_id}", emoji="")
        except Exception as e:
            logger.error("Fact consolidation failed for user %s: %s", user_id, e)

    async def _maybe_learn_instincts(self, user_id: str):
        """Run instinct learning, non-blocking."""
        try:
            await self.instinct_engine.learn_from_recent(user_id)
        except Exception as e:
            logger.warning("Instinct learning failed for user %s: %s", user_id, e)

    async def _maybe_reconcile_fts(self):
        """Reconcile FTS index with ChromaDB — re-index orphans, clean stale entries."""
        now = time.monotonic()
        if now - self._last_fts_reconciliation < 300:
            return
        self._last_fts_reconciliation = now

        try:
            fts_ids = set(await self.backend.fts.get_all_ids())
            chroma_ids = set()
            chroma_entries = {}
            offset = 0
            while True:
                batch = await self.backend.chroma.get_with_embeddings(
                    user_id="", limit=500, offset=offset
                )
                if not batch:
                    break
                for entry in batch:
                    eid = entry["id"]
                    chroma_ids.add(eid)
                    chroma_entries[eid] = entry
                offset += len(batch)
                if len(batch) < 500:
                    break

            orphaned_in_fts = fts_ids - chroma_ids
            for oid in orphaned_in_fts:
                try:
                    await self.backend.fts.delete(oid)
                    logger.info("FTS reconciliation: removed stale entry %s", oid)
                except Exception as e:
                    logger.warning("FTS reconciliation: failed to remove stale %s: %s", oid, e)

            missing_from_fts = chroma_ids - fts_ids
            for mid in missing_from_fts:
                entry = chroma_entries[mid]
                meta = entry.get("metadata") or {}
                content = entry.get("content", "")
                user_id = meta.get("user_id", "")
                level = meta.get("level", "user")
                if not content or not user_id:
                    continue
                try:
                    await self.backend.fts.reindex(mid, content, user_id, level)
                    logger.info("FTS reconciliation: re-indexed %s", mid)
                except Exception as e:
                    logger.warning("FTS reconciliation: failed to re-index %s: %s", mid, e)

            if orphaned_in_fts or missing_from_fts:
                _log_bg("FTS Reconcile", f"{len(orphaned_in_fts)} stale removed, {len(missing_from_fts)} re-indexed", emoji="")
        except Exception as e:
            logger.warning("FTS reconciliation failed: %s", e)

    @staticmethod
    def _recency_score(timestamp_str: str) -> float:
        """Compute recency score (0-1) decaying over ~1 day (half-life ~16.6h)."""
        try:
            created = datetime.fromisoformat(timestamp_str)
            hours_ago = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            return math.exp(-abs(hours_ago) / 24)
        except (ValueError, TypeError, OverflowError):
            return 0.0

    @staticmethod
    def _importance_decay(timestamp_str: str) -> float:
        """Decay multiplier for importance (half-life ~7 days)."""
        try:
            created = datetime.fromisoformat(timestamp_str)
            days_ago = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            if days_ago <= 0:
                return 1.0
            return math.exp(-days_ago / 7)
        except (ValueError, TypeError, OverflowError):
            return 1.0

    def _compute_final_score(self, mem: dict) -> float:
        w_s = self.config.truncation_weight_score
        w_i = self.config.truncation_weight_importance
        w_r = self.config.truncation_weight_recency

        fusion_score = mem.get("score", 0.0)
        raw_importance = mem.get("metadata", {}).get("importance", 3) / 5.0
        decay = self._importance_decay(mem.get("metadata", {}).get("timestamp", ""))
        importance = raw_importance * decay
        recency = self._recency_score(mem.get("metadata", {}).get("timestamp", ""))

        base_score = fusion_score * w_s + importance * w_i + recency * w_r

        tags = mem.get("metadata", {}).get("tags", [])
        if isinstance(tags, list) and "atomic_fact" in tags:
            base_score *= 1.5

        return base_score

    async def search_memory(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = None,
        level_filter: Optional[List[str]] = None,
        workspace_path: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[SearchResult]:
        """Recall with smart truncation and level-aware weighting."""
        user_id = user_id or self.config.default_user_id

        embedding = await get_embedding(query, self.config.embedding_model)
        results = await self.backend.search(
            embedding, user_id, top_k, query_text=query,
            level_filter=level_filter,
        )

        # Hierarchical workspace filter: siblings and children visible, parent invisible
        if workspace_path:
            results = [
                m for m in results
                if self._is_workspace_visible(
                    m.get("metadata", {}).get("workspace_path"), workspace_path
                )
            ]

        # Filter out consolidated, expired, and archived memories
        results = [
            m for m in results
            if not m.get("metadata", {}).get("consolidated")
            and not m.get("metadata", {}).get("expired")
            and not m.get("metadata", {}).get("archived")
        ]

        # Level-weighted scoring when no filter: boost session > user > knowledge
        if not level_filter:
            level_weights = {
                "session": self.config.level_weight_session,
                "user": self.config.level_weight_user,
                "knowledge": self.config.level_weight_knowledge,
            }
            for m in results:
                lvl = m.get("metadata", {}).get("level", "user")
                m["score"] = m["score"] * level_weights.get(lvl, 1.0)

        budget = max_tokens if max_tokens is not None else self.config.token_budget

        scored = [(self._compute_final_score(m), m) for m in results]
        scored.sort(key=lambda x: x[0], reverse=True)

        limited_results = []
        total_tokens = 0
        for _, mem in scored:
            meta_str = str(mem.get("metadata", {}))
            meta_tokens = len(self._tokenizer.encode(meta_str))
            content_tokens = len(self._tokenizer.encode(mem["content"]))
            mem_tokens = meta_tokens + content_tokens
            if mem_tokens <= 0:
                continue
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
        """Soft-delete a memory (mark as archived)."""
        async with self._write_lock:
            success = await self.backend.update_metadata(memory_id, {"archived": True})
        if success:
            logger.info("Memory soft-deleted: %s", memory_id)
        return success

    async def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory (hide from recall, still in storage)."""
        async with self._write_lock:
            success = await self.backend.update_metadata(memory_id, {"archived": True})
        if success:
            logger.info("Memory archived: %s", memory_id)
        return success

    async def unarchive_memory(self, memory_id: str) -> bool:
        """Restore an archived memory."""
        async with self._write_lock:
            success = await self.backend.update_metadata(memory_id, {"archived": False})
        if success:
            logger.info("Memory unarchived: %s", memory_id)
        return success

    async def list_memories(
        self, limit: int = 100, offset: int = 0, user_id: str = None
    ) -> List[MemoryRecord]:
        """List memories for a user (excludes archived)."""
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
            if not m.get("metadata", {}).get("archived")
        ]