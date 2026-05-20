import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from memorymesh.memory.hybrid_backend import HybridBackend


SAMPLE_EMBEDDING = [0.1] * 384


@pytest_asyncio.fixture
async def hybrid(app_config):
    backend = HybridBackend(app_config)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_add_to_both(hybrid):
    mid = await hybrid.add("user1", "Hybrid test", SAMPLE_EMBEDDING)
    assert mid is not None

    vector_results = await hybrid.chroma.search(SAMPLE_EMBEDDING, "user1", top_k=10)
    assert any(r["id"] == mid for r in vector_results)

    fts_results = await hybrid.fts.search("Hybrid", "user1", limit=10)
    assert any(r["id"] == mid for r in fts_results)


@pytest.mark.asyncio
async def test_search_with_query_text(hybrid):
    await hybrid.add("user1", "Hà Nội là thủ đô", SAMPLE_EMBEDDING)
    await hybrid.add("user1", "Python programming", SAMPLE_EMBEDDING)

    results = await hybrid.search(
        SAMPLE_EMBEDDING, "user1", top_k=5, query_text="thủ đô"
    )
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_search_without_query_text_pure_vector(hybrid):
    mid = await hybrid.add("user1", "Pure vector search", SAMPLE_EMBEDDING)

    results = await hybrid.search(
        SAMPLE_EMBEDDING, "user1", top_k=5, query_text=None
    )
    ids = [r["id"] for r in results]
    assert mid in ids


@pytest.mark.asyncio
async def test_delete_from_both(hybrid):
    mid = await hybrid.add("user1", "To delete", SAMPLE_EMBEDDING)

    deleted = await hybrid.delete(mid)
    assert deleted is True

    vector_results = await hybrid.chroma.search(
        SAMPLE_EMBEDDING, "user1", top_k=10
    )
    assert not any(r["id"] == mid for r in vector_results)


@pytest.mark.asyncio
async def test_delete_nonexistent(hybrid):
    result = await hybrid.delete("non-existent-id")
    assert result is True


@pytest.mark.asyncio
async def test_update_metadata(hybrid):
    mid = await hybrid.add("user1", "Update test", SAMPLE_EMBEDDING)

    updated = await hybrid.update_metadata(mid, {"importance": 5})
    assert updated is True

    results = await hybrid.search(SAMPLE_EMBEDDING, "user1", top_k=10)
    for r in results:
        if r["id"] == mid:
            assert r["metadata"]["importance"] == 5
            break


@pytest.mark.asyncio
async def test_list_all(hybrid):
    mids = []
    for i in range(3):
        mid = await hybrid.add("user1", f"List test {i}", SAMPLE_EMBEDDING)
        mids.append(mid)

    results = await hybrid.list_all("user1", limit=10, offset=0)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_fts_failure_fallback(hybrid):
    mid = await hybrid.add("user1", "Fallback test", SAMPLE_EMBEDDING)

    with patch.object(hybrid.fts, "search", new=AsyncMock(side_effect=Exception("FTS down"))):
        results = await hybrid.search(
            SAMPLE_EMBEDDING, "user1", top_k=5, query_text="fallback"
        )
        assert len(results) >= 1
        assert any(r["id"] == mid for r in results)
