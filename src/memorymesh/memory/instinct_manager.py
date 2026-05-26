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

logger = logging.getLogger(__name__)


class CompiledInstinct(NamedTuple):
    instinct_id: int
    project_id: str
    regex: re.Pattern
    reaction: str
    confidence: float


class InstinctManager:
    """In-memory cache of regex-based instincts, grouped by project_id.

    All regex patterns are pre-compiled at load time. Matching runs in O(1)
    per project via dict lookup + compiled regex search (microsecond-level).
    """

    def __init__(self, store: InstinctStore):
        self._store = store
        self._cache: Dict[str, List[CompiledInstinct]] = {}

    async def load_all(self):
        """Load all instincts from DB into RAM cache, pre-compiling regex."""
        self._cache.clear()
        project_ids = await self._store.get_all_projects_v2()
        for pid in project_ids:
            records = await self._store.get_instincts_v2(pid)
            compiled = []
            for row in records:
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
        compiled = []
        for row in records:
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
        confidence_score descending. O(1) lookup + O(N) matching where N is
        the number of instincts for this project (typically < 50).
        """
        instincts = self._cache.get(project_id, [])
        if not instincts:
            return []
        matched = []
        for inst in instincts:
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
    """
    try:
        new_instincts = await asyncio.to_thread(
            InstinctManager.extract_ngrams, tool_sequences, 2
        )
        for inst in new_instincts:
            await store.add_instinct_v2(
                project_id=project_id,
                trigger_regex=inst["trigger_regex"],
                reaction=inst["reaction"],
                confidence_score=inst["confidence_score"],
            )
        if new_instincts:
            await manager.load_project(project_id)
            logger.info(
                "Background learning: added %d new instincts for %s",
                len(new_instincts), project_id,
            )
    except Exception as e:
        logger.error("Background learning daemon failed: %s", e, exc_info=True)
