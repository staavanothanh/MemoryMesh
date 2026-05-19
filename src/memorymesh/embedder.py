"""Singleton SentenceTransformer embedder, async-safe."""

import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

_embedder_instance = None
_model_name = None


def _load_model(name: str) -> SentenceTransformer:
    """Load or reuse the embedding model."""
    global _embedder_instance, _model_name
    if _embedder_instance is None or _model_name != name:
        _embedder_instance = SentenceTransformer(name)
        _model_name = name
    return _embedder_instance


async def get_embedding(text: str, model_name: str) -> List[float]:
    """Compute embedding for text using the cached model (runs in thread)."""
    model = _load_model(model_name)
    embedding = await asyncio.to_thread(model.encode, text)
    return embedding.tolist()