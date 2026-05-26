import pytest

from memorymesh.config import EmbeddingConfig
from memorymesh.embedder import init_embedder, get_embedding
from memorymesh.embeddings.providers import LocalEmbeddingProvider


@pytest.mark.asyncio
async def test_get_embedding_returns_list_of_floats():
    cfg = EmbeddingConfig(mode="local", model="paraphrase-multilingual-MiniLM-L12-v2")
    await init_embedder(cfg)
    result = await get_embedding("Hello world")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_get_embedding_consistent_dimension():
    cfg = EmbeddingConfig(mode="local", model="paraphrase-multilingual-MiniLM-L12-v2")
    await init_embedder(cfg)
    r1 = await get_embedding("Xin chào")
    r2 = await get_embedding("Hello")
    assert len(r1) == len(r2)
    assert len(r1) == 384


@pytest.mark.asyncio
async def test_get_embedding_different_inputs_different_vectors():
    cfg = EmbeddingConfig(mode="local", model="paraphrase-multilingual-MiniLM-L12-v2")
    await init_embedder(cfg)
    r1 = await get_embedding("Cat")
    r2 = await get_embedding("Dog")
    assert r1 != r2


@pytest.mark.asyncio
async def test_local_provider_prewarm():
    provider = LocalEmbeddingProvider("paraphrase-multilingual-MiniLM-L12-v2")
    await provider.prewarm()
    emb = await provider.get_embedding("test")
    assert len(emb) == 384
