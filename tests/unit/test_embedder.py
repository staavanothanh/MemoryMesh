import pytest

from memorymesh.embedder import _load_model, get_embedding


@pytest.mark.asyncio
async def test_get_embedding_returns_list_of_floats():
    result = await get_embedding("Hello world", "paraphrase-multilingual-MiniLM-L12-v2")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_get_embedding_consistent_dimension():
    r1 = await get_embedding("Xin chào", "paraphrase-multilingual-MiniLM-L12-v2")
    r2 = await get_embedding("Hello", "paraphrase-multilingual-MiniLM-L12-v2")
    assert len(r1) == len(r2)
    assert len(r1) == 384


@pytest.mark.asyncio
async def test_get_embedding_different_inputs_different_vectors():
    r1 = await get_embedding("Cat", "paraphrase-multilingual-MiniLM-L12-v2")
    r2 = await get_embedding("Dog", "paraphrase-multilingual-MiniLM-L12-v2")
    assert r1 != r2


def test_load_model_reuses_instance():
    m1 = _load_model("paraphrase-multilingual-MiniLM-L12-v2")
    m2 = _load_model("paraphrase-multilingual-MiniLM-L12-v2")
    assert m1 is m2


def test_load_model_different_name_new_instance():
    original = _load_model("paraphrase-multilingual-MiniLM-L12-v2")
    with pytest.raises(Exception):
        _load_model("non-existent-model")
