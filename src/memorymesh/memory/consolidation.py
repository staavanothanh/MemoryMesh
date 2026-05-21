"""Memory consolidation engine — detect similar memories, merge via LLM."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from ..config import AppConfig
from ..router import RouterClient
from ..embedder import get_embedding

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """You are a memory consolidation assistant. Below are several related memories that may overlap or contain redundant information. Your task is to merge them into a single, concise, information-rich memory.

Memories:
{contents}

Return a JSON object with:
1. "content": the merged summary (maximum 200 words)
2. "tags": merged list of relevant tags (as a JSON array of strings)
3. "importance": importance level 1-5 (integer)

Return ONLY valid JSON, no extra text."""

FACT_CONTRADICTION_PROMPT = """You are a fact contradiction detector. Below is a group of atomic facts about the same topic. Detect if any facts contradict each other.

Rules:
- Two facts contradict if they cannot both be true simultaneously
- "User prefers X" and "User prefers Y" are NOT contradictory (preferences can change)
- "User likes X" and "User dislikes X" ARE contradictory
- "Project uses X" and "Project uses Y" are NOT contradictory (can use both)
- "Project does not use X" and "Project uses X" ARE contradictory
- Facts with newer timestamps override older ones when contradictory
- If no contradiction found, return empty resolutions list

Return a JSON object:
{
  "contradictions_found": true/false,
  "resolutions": [
    {
      "keep": "exact text of the fact to keep",
      "remove": ["exact text of facts to remove"],
      "reason": "why this resolution"
    }
  ]
}

Return ONLY valid JSON, no extra text.

Facts:
{facts}

Output:"""


class ConsolidationEngine:
    """Find clusters of similar memories and merge them via LLM."""

    def __init__(self, config: AppConfig, backend, router: RouterClient):
        self.config = config
        self.backend = backend
        self.router = router
        self.threshold = config.consolidation.similarity_threshold
        self.min_cluster_size = config.consolidation.min_cluster_size
        self.batch_size = config.consolidation.batch_size
        self._fact_threshold = 0.7  # lower threshold for grouping facts by topic

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na * nb == 0:
            return 0.0
        return dot / (na * nb)

    def _find_clusters(
        self, memories: List[Dict[str, Any]], threshold: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        t = threshold if threshold is not None else self.threshold
        n = len(memories)
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = self._cosine_sim(memories[i]["embedding"], memories[j]["embedding"])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            cluster = [memories[i]]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and sim_matrix[i][j] >= t:
                    cluster.append(memories[j])
                    visited[j] = True
            clusters.append(cluster)
        return clusters

    async def _merge_cluster(self, cluster: List[Dict[str, Any]], user_id: str):
        try:
            contents = "\n---\n".join(
                f"[{m['metadata'].get('importance', 3)}] {m['content']}"
                for m in cluster
            )
            prompt = CONSOLIDATION_PROMPT.format(contents=contents)
            response = await self.router.call_llm(prompt)
            meta = json.loads(response)

            new_content = meta.get("content", cluster[0]["content"])
            new_tags = meta.get("tags", [])
            new_importance = meta.get("importance", 3)

            embedding = await get_embedding(new_content, self.config.embedding_model)

            merged_id = await self.backend.add(
                user_id=user_id,
                content=new_content,
                embedding=embedding,
                metadata={
                    "importance": new_importance,
                    "tags": new_tags,
                    "merged": True,
                    "level": "knowledge",
                },
                level="knowledge",
            )

            for mem in cluster:
                try:
                    await self.backend.update_metadata(mem["id"], {
                        "consolidated": True,
                        "merged_into": merged_id,
                    })
                except Exception as e:
                    logger.warning("Failed to mark %s as consolidated: %s", mem["id"], e)

            logger.info(
                "Merged %d memories into %s for user %s",
                len(cluster), merged_id, user_id
            )

        except Exception as e:
            logger.error("Cluster merge failed: %s", e)

    async def _resolve_fact_contradictions(
        self, group: List[Dict[str, Any]], user_id: str
    ):
        """Check a group of related facts for contradictions and resolve via LLM."""
        if len(group) < 2:
            return
        try:
            facts_text = "\n".join(
                f"[{m['metadata'].get('timestamp', 'unknown')}] {m['content']}"
                for m in group
            )
            prompt = FACT_CONTRADICTION_PROMPT.format(facts=facts_text)
            response = await self.router.call_llm(prompt)
            data = json.loads(response)
            if not data.get("contradictions_found"):
                return
            for resolution in data.get("resolutions", []):
                keep_text = resolution.get("keep", "")
                remove_texts = resolution.get("remove", [])
                if not keep_text or not remove_texts:
                    continue
                keep_mem = None
                for m in group:
                    if m["content"] == keep_text:
                        keep_mem = m
                        break
                if not keep_mem:
                    continue
                for m in group:
                    if m["content"] in remove_texts:
                        try:
                            await self.backend.update_metadata(m["id"], {
                                "fact_resolved": True,
                                "resolved_by": keep_mem["id"],
                                "resolution_reason": resolution.get("reason", ""),
                            })
                            logger.info(
                                "Fact contradiction resolved: '%s' overrides '%s' — %s",
                                keep_text, m["content"], resolution.get("reason", "")
                            )
                        except Exception as e:
                            logger.warning("Failed to mark fact %s as resolved: %s", m["id"], e)
        except Exception as e:
            logger.error("Fact contradiction resolution failed: %s", e)

    async def run_fact_consolidation(self, user_id: str) -> int:
        """Resolve contradictory atomic facts. Returns number of resolutions applied."""
        if not self.config.consolidation.enabled:
            return 0
        all_mems = await self.backend.get_with_embeddings(user_id, limit=self.batch_size)
        active_facts = [
            m for m in all_mems
            if "atomic_fact" in m["metadata"].get("tags", [])
            and not m["metadata"].get("fact_resolved")
        ]
        if len(active_facts) < 2:
            return 0
        groups = await asyncio.to_thread(self._find_clusters, active_facts, self._fact_threshold)
        resolved_count = 0
        for group in groups:
            if len(group) >= 2:
                await self._resolve_fact_contradictions(group, user_id)
                resolved_count += 1
        if resolved_count:
            logger.info("Fact consolidation: resolved %d groups for user %s", resolved_count, user_id)
        return resolved_count

    async def run_for_user(self, user_id: str) -> int:
        """Run one consolidation pass. Returns number of merges performed."""
        if not self.config.consolidation.enabled:
            return 0

        all_mems = await self.backend.get_with_embeddings(
            user_id, limit=self.batch_size
        )
        active = [m for m in all_mems if not m["metadata"].get("consolidated")]
        if len(active) < self.min_cluster_size:
            return 0

        clusters = await asyncio.to_thread(self._find_clusters, active)
        merged_count = 0
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                await self._merge_cluster(cluster, user_id)
                merged_count += 1

        return merged_count
