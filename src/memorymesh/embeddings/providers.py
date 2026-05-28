"""Embedding providers — local SentenceTransformer and remote API."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
import numpy as np

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)

DIM = 384


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        ...

    async def get_dimension(self) -> int:
        """Return the embedding dimension. Default: call get_embedding on a ping string."""
        emb = await self.get_embedding("ping")
        return len(emb)

    @abstractmethod
    async def prewarm(self):
        ...

    async def close(self):
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """SentenceTransformer-based local embedding provider (requires memorymesh[local])."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    async def get_embedding(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError("LocalEmbeddingProvider not initialized. Call prewarm() first.")
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    async def get_dimension(self) -> int:
        if self._dimension is None:
            emb = await self.get_embedding("ping")
            self._dimension = len(emb)
        return self._dimension

    async def prewarm(self):
        try:
            from sentence_transformers import SentenceTransformer
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self._model_name)
            )
            ping_emb = await self.get_embedding("ping")
            self._dimension = len(ping_emb)
            logger.info("LocalEmbeddingProvider ready (model=%s, dim=%d)", self._model_name, self._dimension)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install 'memorymesh[local]'"
            )


class RemoteEmbeddingProvider(EmbeddingProvider):
    """Remote API-based embedding provider (lightweight, no local model)."""

    def __init__(self, config: EmbeddingConfig):
        self._api_url = config.remote_api_url.rstrip("/") + "/embeddings" if config.remote_api_url else ""
        self._api_key = config.remote_api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._dimension: Optional[int] = None

    async def get_embedding(self, text: str) -> List[float]:
        if not self._client:
            raise RuntimeError("RemoteEmbeddingProvider not initialized. Call prewarm() first.")
        try:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            payload = {"input": text, "model": "text-embedding-ada-002"}
            response = await self._client.post(
                self._api_url, json=payload, headers=headers, timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            emb = data.get("data", [{}])[0].get("embedding", None)
            if emb is None:
                emb = data.get("embedding", [0.0] * DIM)
            return emb
        except Exception as e:
            logger.error("Remote embedding failed: %s", e)
            raise

    async def get_dimension(self) -> int:
        if self._dimension is None:
            emb = await self.get_embedding("ping")
            self._dimension = len(emb)
        return self._dimension

    async def prewarm(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        try:
            ping_emb = await self.get_embedding("ping")
            self._dimension = len(ping_emb)
            logger.info("RemoteEmbeddingProvider ready (url=%s, dim=%d)", self._api_url, self._dimension)
        except Exception as e:
            logger.warning("Remote embedding prewarm ping failed (will retry): %s", e)
            self._dimension = DIM

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class NoneEmbeddingProvider(EmbeddingProvider):
    """FTS-only mode — no embedding computation needed."""
    DIM = 384

    async def get_embedding(self, text: str) -> List[float]:
        return [0.0] * self.DIM

    async def get_dimension(self) -> int:
        return self.DIM

    async def prewarm(self):
        pass
