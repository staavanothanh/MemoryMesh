import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

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
from ..utils.json_parser import clean_and_parse_llm_json

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

        # FTS reconciliation no longer needed — vector + FTS in single ACID transaction

    async def _enrich_memory(self, memory_id: str, text: str, user_id: str):
        """Call LLM to get tags, importance, summary and merge with existing metadata."""
        try:
            existing = await self.backend._get_by_id_full(memory_id)
            if not existing:
                return
            existing_meta = existing["metadata"]
            existing_tags = set(existing_meta.get("tags", []) or [])
            existing_imp = existing_meta.get("importance", 3)

            prompt = EXTRACT_METADATA_PROMPT.format(content=text)
            response = await self.router.call_llm_background(prompt, json_mode=True)
            meta = clean_and_parse_llm_json(response)
            update = {}
            if "tags" in meta and isinstance(meta["tags"], list):
                merged_tags = list(existing_tags | set(meta["tags"]))
                update["tags"] = merged_tags
            if "importance" in meta and isinstance(meta["importance"], int):
                update["importance"] = max(existing_imp, meta["importance"])
            if "summary" in meta and isinstance(meta["summary"], str):
                update["summary"] = meta["summary"]
            if update:
                await self.backend.update_metadata(memory_id, update)
                logger.info("Enrichment merged for %s: %s", memory_id, update)
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

    @staticmethod
    def _extract_query_keywords(query: str) -> str:
        """Extract content words from query for FTS, drop question words."""
        QUESTION_WORDS = {
            "cuoi", "buoi", "thao", "luan", "gi", "la", "co", "the",
            "nao", "nhu", "khi", "sau", "truoc", "ban", "ve",
            "what", "when", "how", "did", "was", "were", "the",
            "last", "latest", "recent", "discuss", "discussed",
            "nao", "nhe", "nhe", "ak", "ah", "a", "o", "di", "nhe",
            "guong", "ma", "roi", "thu", "nha", "khong",
            "continue", "continues", "continued",
        }
        import re
        words = re.findall(r"[a-zA-ZÀ-ỹ]+", query.lower())
        meaningful = [w for w in words if w not in QUESTION_WORDS and len(w) >= 3]
        return " ".join(meaningful[:5])

    async def _enrich_fts_results(
        self,
        fts_results: List[Dict],
        user_id: str,
    ) -> List[Dict]:
        """Attach metadata from backend to FTS results for scoring and filtering."""
        if not fts_results:
            return []
        ids = [r["id"] for r in fts_results]
        try:
            chroma_data = await self.backend.get_with_embeddings_by_ids(ids)
            chroma_map = {d["id"]: d for d in chroma_data}
            enriched = []
            for r in fts_results:
                chroma = chroma_map.get(r["id"], {})
                meta = chroma.get("metadata", {}) or {}
                enriched.append({
                    **r,
                    "metadata": meta,
                    "score": r.get("score", 0.0),
                })
            return enriched
        except Exception:
            return [{**r, "metadata": {}} for r in fts_results]

    async def search_with_fallback(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = None,
        workspace_path: Optional[str] = None,
        max_tokens: Optional[int] = None,
        min_score_threshold: float = 0.25,
    ) -> tuple:
        """3-tier fallback retrieval. Returns (results, tier_name, metadata)."""
        uid = user_id or self.config.default_user_id

        # Tier 1: Semantic search via search_memory (with existing filters, budget)
        tier1 = await self.search_memory(
            query=query, top_k=top_k, user_id=uid,
            workspace_path=workspace_path, max_tokens=max_tokens,
        )
        tier1_filtered = [r for r in tier1 if r.get("score", 0) >= min_score_threshold]
        if tier1_filtered:
            return tier1_filtered[:top_k], "semantic", {}

        # Tier 2: FTS keyword search
        keywords = self._extract_query_keywords(query)
        if keywords:
            try:
                tier2_raw = await self.backend.fts_search(
                    keywords, uid, limit=top_k * 2,
                )
                tier2_enriched = await self._enrich_fts_results(tier2_raw, uid)
                # Filter out archived/consolidated/expired
                tier2_filtered = [
                    r for r in tier2_enriched
                    if not r.get("metadata", {}).get("archived")
                    and not r.get("metadata", {}).get("consolidated")
                    and not r.get("metadata", {}).get("expired")
                ]
                # Apply workspace filter
                if workspace_path:
                    tier2_filtered = [
                        r for r in tier2_filtered
                        if self._is_workspace_visible(
                            r.get("metadata", {}).get("workspace_path"), workspace_path
                        )
                    ]
                if tier2_filtered:
                    # Convert enriched dict to SearchResult
                    results = self._dicts_to_search_results(tier2_filtered[:top_k])
                    return results, "fts_keyword", {}
            except Exception as e:
                logger.warning("Tier 2 FTS failed: %s", e)

        # Tier 3: Chronological scan — always has results
        try:
            tier3_raw = await self.backend.list_recent(uid, limit=top_k)
            tier3_enriched = await self._enrich_fts_results(tier3_raw, uid)
            # Apply filters
            tier3_filtered = [
                r for r in tier3_enriched
                if not r.get("metadata", {}).get("archived")
                and not r.get("metadata", {}).get("consolidated")
                and not r.get("metadata", {}).get("expired")
            ]
            if workspace_path:
                tier3_filtered = [
                    r for r in tier3_filtered
                    if self._is_workspace_visible(
                        r.get("metadata", {}).get("workspace_path"), workspace_path
                    )
                ]
            if tier3_filtered:
                results = self._dicts_to_search_results(tier3_filtered[:top_k])
                return results, "chronological", {}
        except Exception as e:
            logger.warning("Tier 3 chronological fallback failed: %s", e)

        return [], "empty", {}

    def _dicts_to_search_results(self, dicts: List[Dict]) -> List[SearchResult]:
        """Convert internal dict results to SearchResult objects."""
        return [
            SearchResult(
                id=d.get("id", ""),
                content=d.get("content", ""),
                score=d.get("score", 0.0),
                tags=d.get("metadata", {}).get("tags", []),
                importance=d.get("metadata", {}).get("importance", 3),
                timestamp=d.get("metadata", {}).get("timestamp", ""),
            )
            for d in dicts
        ]

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

    async def delete_memories_by_session(self, session_id: str, user_id: str = "") -> int:
        tag = f"session:{session_id}"
        if hasattr(self.backend, "delete_by_tag"):
            uid = user_id or self.config.default_user_id
            return await self.backend.delete_by_tag(uid, tag)
        return 0

    async def preserve_important_memories(self, session_id: str, user_id: str = "") -> int:
        """Copy important session memories to knowledge level before cascade delete."""
        tag = f"session:{session_id}"
        if not hasattr(self.backend, "list_by_tag"):
            return 0
        uid = user_id or self.config.default_user_id
        memories = await self.backend.list_by_tag(uid, tag)
        KEYWORDS = {
            "plan", "kế hoạch", "architecture", "design decision",
            "quyết định", "thiết kế", "next step", "tiếp theo",
            "refactor", "implement", "fix", "thay đổi",
            "milestone", "goal", "mục tiêu",
        }
        preserved = 0
        for mem in memories:
            meta = mem.get("metadata", {})
            imp = mem.get("importance", 3)
            content = mem.get("content", "")
            if imp >= 4 or any(kw in content.lower() for kw in KEYWORDS):
                await self.add_memory(
                    text=content,
                    tags=["preserved", "session_summary", tag],
                    importance=min(imp + 1, 5),
                    level="knowledge",
                    user_id=mem["user_id"],
                    workspace_path=meta.get("workspace_path"),
                )
                preserved += 1
        if preserved:
            logger.info("Preserved %d important memories from session %s", preserved, session_id[:8])
        return preserved

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
        """List memories for a user (archived filter pushed to SQL)."""
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