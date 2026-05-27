"""InstinctManager — RAM cache of compiled regex instincts for O(1) lookup.

Pre-compiles regex patterns at load time for microsecond-level matching.
Hot-reloads when new instincts are learned.
"""

import re
import logging
import asyncio
from collections import Counter
from typing import Dict, List, Optional, NamedTuple

from .instinct_store import InstinctStore
from ..config import InstinctConfig

logger = logging.getLogger(__name__)


class CompiledInstinct(NamedTuple):
    instinct_id: int
    project_id: str
    regex: re.Pattern
    reaction: str
    confidence: float


# ReDoS: catastrophic backtracking patterns (nested quantifiers, overlapping groups)
_CATASTROPHIC_PATTERNS = re.compile(
    r"\(.+\)\+"        # (something)+ — nested group with quantifier
    r"|\(\?.+\)\+"      # (?...) non-capturing group with quantifier
    r"|\+.+"            # quantified after quantifier
    r"|\*\+"            # star followed by plus
    r"|\([^)]+\)\*"     # group with star
)


class InstinctManager:
    """In-memory cache of regex-based instincts, grouped by project_id.

    All regex patterns are pre-compiled at load time. Matching runs in O(1)
    per project via dict lookup + compiled regex search (microsecond-level).
    """

    def __init__(self, store: InstinctStore, config: Optional[InstinctConfig] = None):
        self._store = store
        self._config = config or InstinctConfig()
        self._cache: Dict[str, List[CompiledInstinct]] = {}

    @staticmethod
    def is_safe_regex(pattern: str, max_length: int = 200) -> bool:
        """ReDoS guard: reject patterns that are too long or have catastrophic backtracking."""
        if len(pattern) > max_length:
            logger.debug("ReDoS guard: pattern too long (%d > %d)", len(pattern), max_length)
            return False
        if _CATASTROPHIC_PATTERNS.search(pattern):
            logger.debug("ReDoS guard: catastrophic pattern detected: %s", pattern[:80])
            return False
        return True

    @staticmethod
    def _dedup_instincts(records: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """Dedup instincts with the same trigger_regex (keep highest confidence)."""
        seen: Dict[str, Dict] = {}
        for row in records:
            key = row["trigger_regex"].strip().lower()
            existing = seen.get(key)
            if existing is None or row.get("confidence_score", 0) > existing.get("confidence_score", 0):
                seen[key] = row
        return list(seen.values())

    async def load_all(self):
        """Load all instincts from DB into RAM cache, pre-compiling regex."""
        self._cache.clear()
        project_ids = await self._store.get_all_projects_v2()
        for pid in project_ids:
            records = await self._store.get_instincts_v2(pid)
            records = self._dedup_instincts(records, self._config.dedup_similarity_threshold)
            compiled = []
            for row in records:
                if not self.is_safe_regex(row["trigger_regex"], self._config.max_pattern_length):
                    continue
                try:
                    pattern = re.compile(row["trigger_regex"], re.IGNORECASE)
                    compiled.append(CompiledInstinct(
                        instinct_id=row["id"],
                        project_id=row["project_id"],
                        regex=pattern,
                        reaction=row["reaction"],
                        confidence=row["confidence_score"],
                    ))
                except re.error:
                    logger.debug("Skipping invalid regex for instinct %s", row["id"])
            compiled.sort(key=lambda x: x.confidence, reverse=True)
            self._cache[pid] = compiled
        logger.info("InstinctManager loaded %d project(s) into RAM cache", len(self._cache))

    async def load_project(self, project_id: str):
        """Hot-reload a specific project's instincts into RAM."""
        records = await self._store.get_instincts_v2(project_id)
        records = self._dedup_instincts(records, self._config.dedup_similarity_threshold)
        compiled = []
        for row in records:
            if not self.is_safe_regex(row["trigger_regex"], self._config.max_pattern_length):
                continue
            try:
                pattern = re.compile(row["trigger_regex"], re.IGNORECASE)
                compiled.append(CompiledInstinct(
                    instinct_id=row["id"],
                    project_id=row["project_id"],
                    regex=pattern,
                    reaction=row["reaction"],
                    confidence=row["confidence_score"],
                ))
            except re.error:
                continue
        compiled.sort(key=lambda x: x.confidence, reverse=True)
        if compiled:
            self._cache[project_id] = compiled
        elif project_id in self._cache:
            del self._cache[project_id]

    def evaluate(self, project_id: str, context_text: str) -> List[str]:
        """Match context_text against all compiled instincts for a project.

        Returns list of reaction strings for matching instincts, ordered by
        confidence_score descending. Applies confidence_floor filter.
        O(1) lookup + O(N) matching where N is the number of instincts.
        """
        instincts = self._cache.get(project_id, [])
        if not instincts:
            return []
        matched = []
        for inst in instincts:
            if inst.confidence < self._config.confidence_floor:
                continue
            if inst.regex.search(context_text):
                matched.append(inst.reaction)
        return matched

    def clear(self):
        self._cache.clear()

    @staticmethod
    def extract_ngrams(tool_sequences: List[str], threshold: int = 2) -> list:
        """CPU-bound N-gram extraction — run via asyncio.to_thread.

        Analyzes a list of tool action strings to find frequent N-gram
        sequences (length 2-5) that appear >= threshold times.
        Returns list of dicts: {trigger_regex, reaction, confidence_score}.
        """
        ngrams: Counter = Counter()
        for seq in tool_sequences:
            tokens = seq.strip().split()
            for n in range(2, min(6, len(tokens) + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i:i + n])
                    ngrams[ngram] += 1

        new_instincts = []
        for ngram_text, count in ngrams.items():
            if count < threshold:
                continue
            tokens = ngram_text.split()
            action_regex = r".*".join(re.escape(t) for t in tokens)
            confidence = min(0.9, 0.2 + count * 0.15)
            reaction = (
                f"[Instinct] Detected frequent workflow: '{ngram_text}' "
                f"(appeared {count}x). Consider automating this sequence."
            )
            new_instincts.append({
                "trigger_regex": action_regex,
                "reaction": reaction,
                "confidence_score": round(confidence, 4),
            })
        return new_instincts


async def background_learning_daemon(
    manager: InstinctManager,
    store: InstinctStore,
    tool_sequences: List[str],
    project_id: str,
):
    """Background task: extract N-gram patterns and store + hot-reload.

    Runs N-gram extraction in a thread pool to avoid blocking the event loop.
    Applies ReDoS guard and cap enforcement before inserting each instinct.
    """
    try:
        new_instincts = await asyncio.to_thread(
            InstinctManager.extract_ngrams, tool_sequences, 2
        )
        added = 0
        for inst in new_instincts:
            if not manager.is_safe_regex(inst["trigger_regex"], manager._config.max_pattern_length):
                logger.debug("Background learning: skipping unsafe regex for %s", project_id)
                continue
            result = await store.add_instinct_v2(
                project_id=project_id,
                trigger_regex=inst["trigger_regex"],
                reaction=inst["reaction"],
                confidence_score=inst["confidence_score"],
            )
            if result is not None:
                added += 1
        if added:
            await manager.load_project(project_id)
            logger.info(
                "Background learning: added %d new instincts for %s",
                added, project_id,
            )
    except Exception as e:
        logger.error("Background learning daemon failed: %s", e, exc_info=True)
