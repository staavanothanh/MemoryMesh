import pytest
from memorymesh.memory.chroma_impl import ChromaMemoryBackend
from memorymesh.config import ChromaConfig


@pytest.fixture
def backend(chroma_config):
    return ChromaMemoryBackend(chroma_config.db_path)


@pytest.fixture
def sample_embedding():
    return [0.1] * 384


@pytest.mark.asyncio
async def test_add_and_search(backend, sample_embedding):
    memory_id = await backend.add(
        user_id="test_user",
        content="Hà Nội là thủ đô của Việt Nam",
        embedding=sample_embedding,
        metadata={"tags": ["địa lý"], "importance": 3},
    )
    assert memory_id is not None
    assert isinstance(memory_id, str)

    results = await backend.search(sample_embedding, "test_user", top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == memory_id
    assert results[0]["content"] == "Hà Nội là thủ đô của Việt Nam"
    assert results[0]["metadata"]["tags"] == ["địa lý"]


@pytest.mark.asyncio
async def test_search_filters_by_user(backend, sample_embedding):
    await backend.add(
        user_id="user_a",
        content="Alice's memory",
        embedding=sample_embedding,
    )
    await backend.add(
        user_id="user_b",
        content="Bob's memory",
        embedding=sample_embedding,
    )

    results_a = await backend.search(sample_embedding, "user_a", top_k=10)
    results_b = await backend.search(sample_embedding, "user_b", top_k=10)

    assert all(r["metadata"]["user_id"] == "user_a" for r in results_a)
    assert all(r["metadata"]["user_id"] == "user_b" for r in results_b)


@pytest.mark.asyncio
async def test_delete(backend, sample_embedding):
    memory_id = await backend.add(
        user_id="test_user",
        content="To be deleted",
        embedding=sample_embedding,
    )

    deleted = await backend.delete(memory_id)
    assert deleted is True

    results = await backend.search(sample_embedding, "test_user", top_k=10)
    ids = [r["id"] for r in results]
    assert memory_id not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent(backend):
    # ChromaDB silently ignores non-existent IDs
    deleted = await backend.delete("non-existent-id")
    assert deleted is True


@pytest.mark.asyncio
async def test_list_all(backend, sample_embedding):
    ids = []
    for i in range(5):
        mid = await backend.add(
            user_id="test_user",
            content=f"Memory {i}",
            embedding=sample_embedding,
        )
        ids.append(mid)

    results = await backend.list_all("test_user", limit=10, offset=0)
    assert len(results) == 5
    assert all(r["metadata"]["user_id"] == "test_user" for r in results)


@pytest.mark.asyncio
async def test_list_all_pagination(backend, sample_embedding):
    for i in range(5):
        await backend.add(
            user_id="test_user",
            content=f"Memory {i}",
            embedding=sample_embedding,
        )

    page1 = await backend.list_all("test_user", limit=2, offset=0)
    page2 = await backend.list_all("test_user", limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0]["id"] != page2[0]["id"]


@pytest.mark.asyncio
async def test_audit_log_created(backend, sample_embedding):
    memory_id = await backend.add(
        user_id="test_user",
        content="Audit test",
        embedding=sample_embedding,
    )
    audit_count = backend.audit.count()
    assert audit_count >= 1


@pytest.mark.asyncio
async def test_update_metadata_merge(backend, sample_embedding):
    memory_id = await backend.add(
        user_id="test_user",
        content="Update test",
        embedding=sample_embedding,
        metadata={"tags": ["original"], "importance": 2},
    )

    updated = await backend.update_metadata(memory_id, {"importance": 5, "summary": "Test summary"})
    assert updated is True

    results = await backend.search(sample_embedding, "test_user", top_k=10)
    found = next(r for r in results if r["id"] == memory_id)
    assert found["metadata"]["tags"] == ["original"]
    assert found["metadata"]["importance"] == 5
    assert found["metadata"]["summary"] == "Test summary"


@pytest.mark.asyncio
async def test_update_metadata_nonexistent(backend):
    updated = await backend.update_metadata("non-existent-id", {"importance": 5})
    assert updated is False
