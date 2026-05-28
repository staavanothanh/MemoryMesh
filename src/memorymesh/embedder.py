"""Embedding interface — delegates to factory-chosen provider.

Lazy initialization: if init_embedder() hasn't been called,
get_embedding() will auto-init with a local provider for backward compatibility.
"""

import logging
from typing import List, Optional

from .config import EmbeddingConfig
from .embeddings.factory import create_embedding_provider
from .embeddings.providers import EmbeddingProvider

logger = logging.getLogger(__name__)

_provider: Optional[EmbeddingProvider] = None


async def get_embedding(text: str, model_name: str = "") -> List[float]:
    """Compute embedding for text using the configured provider.

    Auto-initializes with local provider if not yet initialized,
    for backward compatibility with tests and existing code.
    """
    global _provider
    if _provider is None:
        cfg = EmbeddingConfig(mode="local", model=model_name or "paraphrase-multilingual-MiniLM-L12-v2")
        await init_embedder(cfg)
    return await _provider.get_embedding(text)


async def init_embedder(config: EmbeddingConfig):
    """Initialize the embedding provider from config."""
    global _provider
    if _provider is not None:
        return
    _provider = create_embedding_provider(config)
    await _provider.prewarm()
    logger.info("Embedder initialized (mode=%s)", config.mode)


async def prewarm_embedder(model_name: str):
    """Backward-compatible prewarm — loads provider with default local config."""
    if _provider is None:
        cfg = EmbeddingConfig(mode="local", model=model_name)
        await init_embedder(cfg)


async def get_embedding_dimension() -> int:
    """Return the embedding dimension of the current provider."""
    global _provider
    if _provider is None:
        cfg = EmbeddingConfig(mode="local", model="paraphrase-multilingual-MiniLM-L12-v2")
        await init_embedder(cfg)
    return await _provider.get_dimension()


async def close_embedder():
    global _provider
    if _provider:
        await _provider.close()
        _provider = None
