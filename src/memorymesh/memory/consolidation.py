"""Memory consolidation engine — detect similar memories, merge via LLM."""

import json
import logging
from typing import List, Dict, Any

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


class ConsolidationEngine:
    """Find clusters of similar memories and merge them via LLM."""

    def __init__(self, config: AppConfig, backend, router: RouterClient):
        self.config = config
        self.backend = backend
        self.router = router
        self.threshold = config.consolidation.similarity_threshold
        self.min_cluster_size = config.consolidation.min_cluster_size
        self.batch_size = config.consolidation.batch_size

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na * nb == 0:
            return 0.0
        return dot / (na * nb)

    def _find_clusters(
        self, memories: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
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
                if not visited[j] and sim_matrix[i][j] >= self.threshold:
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
                },
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

        clusters = self._find_clusters(active)
        merged_count = 0
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                await self._merge_cluster(cluster, user_id)
                merged_count += 1

        return merged_count
