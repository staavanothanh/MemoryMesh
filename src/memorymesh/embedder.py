"""Singleton SentenceTransformer embedder, async-safe with thread-based loading."""

import asyncio
import threading
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

_embedder_instance = None
_model_name = None
_init_lock = threading.Lock()


def _load_model(name: str) -> SentenceTransformer:
    """Load or reuse the embedding model (thread-safe)."""
    global _embedder_instance, _model_name
    if _embedder_instance is not None and _model_name == name:
        return _embedder_instance
    with _init_lock:
        if _embedder_instance is None or _model_name != name:
            _embedder_instance = SentenceTransformer(name)
            _model_name = name
    return _embedder_instance


def _sync_compute(text: str, model_name: str) -> List[float]:
    """Synchronous embedding computation with LRU cache."""
    model = _load_model(model_name)
    return model.encode(text).tolist()


_cached_compute = lru_cache(maxsize=64)(_sync_compute)


async def get_embedding(text: str, model_name: str) -> List[float]:
    """Compute embedding for text using the cached model (runs in thread, cached by text)."""
    return await asyncio.to_thread(_cached_compute, text, model_name)


async def prewarm_embedder(model_name: str):
    """Pre-warm: load model and compute a dummy embedding at startup."""
    await get_embedding("ping", model_name)
