"""Test uncovered paths in sqlite_vec_backend.py:
soft_delete, update_embedding, get_metadata_by_ids, get_with_embeddings_by_ids,
update_metadata error paths, _flag_memories, list_all with pagination."""

import pytest
from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend, normalize_l2, load_vec_extension

SAMPLE_EMBEDDING = [0.1] * 384


@pytest.mark.asyncio
async def test_add_and_lookup(backend: SqliteVecBackend):
    """Test basic add and then lookup via _get_by_id."""
    mem_id = await backend.add(
        user_id="test_user",
        content="Hello world",
        embedding=SAMPLE_EMBEDDING,
        metadata={"importance": 3},
    )
    result = await backend._get_by_id(mem_id)
    assert result is not None
    assert result["content"] == "Hello world"
    assert result["metadata"]["importance"] == 3


@pytest.mark.asyncio
async def test_get_by_id_not_found(backend: SqliteVecBackend):
    """_get_by_id returns None for non-existent ID."""
    result = await backend._get_by_id("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_update_metadata_on_existing_memory(backend: SqliteVecBackend):
    """update_metadata merges new fields into existing metadata."""
    mem_id = await backend.add(
        user_id="test_user",
        content="Test update metadata",
        embedding=SAMPLE_EMBEDDING,
        metadata={"importance": 3, "level": "user"},
    )
    success = await backend.update_metadata(mem_id, {"importance": 5, "tags": ["test"]})
    assert success is True
    result = await backend._get_by_id(mem_id)
    assert result["metadata"]["importance"] == 5
    assert result["metadata"]["tags"] == ["test"]


@pytest.mark.asyncio
async def test_update_metadata_nonexistent_memory(backend: SqliteVecBackend):
    """update_metadata returns False for non-existent memory."""
    success = await backend.update_metadata("nonexistent-id", {"importance": 5})
    assert success is False


@pytest.mark.asyncio
async def test_soft_delete(backend: SqliteVecBackend):
    """soft_delete marks memory as deleted."""
    mem_id = await backend.add(
        user_id="test_user",
        content="To delete",
        embedding=SAMPLE_EMBEDDING,
    )
    success = await backend.soft_delete(mem_id)
    assert success is True
    # After soft delete, should not appear in search results
    results = await backend.list_all("test_user", limit=100)
    ids = [r["id"] for r in results]
    assert mem_id not in ids


@pytest.mark.asyncio
async def test_soft_delete_nonexistent(backend: SqliteVecBackend):
    """soft_delete on non-existent memory returns False.
    
    Note: Must add at least one memory first so the vec_memories virtual table
    is created (lazy creation on first add). The trigger trg_archive_cleanup
    references vec_memories."""
    # Add a memory first to trigger vec_memories creation
    await backend.add(
        user_id="test_user",
        content="Seed memory",
        embedding=SAMPLE_EMBEDDING,
    )
    # Try to soft_delete a different random ID that doesn't exist
    import uuid
    success = await backend.soft_delete(str(uuid.uuid4()))
    assert success is False


@pytest.mark.asyncio
async def test_soft_delete_twice(backend: SqliteVecBackend):
    """soft_delete on already-deleted memory returns False."""
    mem_id = await backend.add(
        user_id="test_user",
        content="Delete twice",
        embedding=SAMPLE_EMBEDDING,
    )
    success = await backend.soft_delete(mem_id)
    assert success is True
    # Second delete should return False (already deleted)
    success = await backend.soft_delete(mem_id)
    assert success is False


@pytest.mark.asyncio
async def test_update_embedding(backend: SqliteVecBackend):
    """update_embedding replaces vector for existing memory."""
    mem_id = await backend.add(
        user_id="test_user",
        content="Update embedding",
        embedding=SAMPLE_EMBEDDING,
    )
    new_embedding = [0.9] * 384
    success = await backend.update_embedding(mem_id, new_embedding)
    assert success is True

    # Verify by searching for the new vector — should find the memory
    results = await backend.search(new_embedding, "test_user", top_k=5)
    ids = [r["id"] for r in results]
    assert mem_id in ids


@pytest.mark.asyncio
async def test_get_metadata_by_ids(backend: SqliteVecBackend):
    """get_metadata_by_ids returns metadata for valid IDs."""
    ids = []
    for i in range(3):
        mid = await backend.add(
            user_id="test_user",
            content=f"Metadata test {i}",
            embedding=SAMPLE_EMBEDDING,
            metadata={"index": i},
        )
        ids.append(mid)

    results = await backend.get_metadata_by_ids(ids)
    assert len(results) == 3
    assert {r["id"] for r in results} == set(ids)
    assert all("content" in r for r in results)


@pytest.mark.asyncio
async def test_get_metadata_by_ids_empty(backend: SqliteVecBackend):
    """get_metadata_by_ids returns empty list for empty input."""
    results = await backend.get_metadata_by_ids([])
    assert results == []


@pytest.mark.asyncio
async def test_get_metadata_by_ids_mixed(backend: SqliteVecBackend):
    """get_metadata_by_ids handles mix of valid and invalid IDs."""
    valid_id = await backend.add(
        user_id="test_user",
        content="Valid memory",
        embedding=SAMPLE_EMBEDDING,
    )
    results = await backend.get_metadata_by_ids([valid_id, "nonexistent-id"])
    assert len(results) == 1
    assert results[0]["id"] == valid_id


@pytest.mark.asyncio
async def test_get_with_embeddings_by_ids(backend: SqliteVecBackend):
    """get_with_embeddings_by_ids returns data with embeddings."""
    ids = []
    for i in range(2):
        mid = await backend.add(
            user_id="test_user",
            content=f"With emb {i}",
            embedding=[0.1 + i * 0.01] * 384,
        )
        ids.append(mid)

    results = await backend.get_with_embeddings_by_ids(ids)
    assert len(results) == 2
    assert all("embedding" in r for r in results)


@pytest.mark.asyncio
async def test_get_with_embeddings_by_ids_empty(backend: SqliteVecBackend):
    """get_with_embeddings_by_ids returns empty list for empty input."""
    results = await backend.get_with_embeddings_by_ids([])
    assert results == []


@pytest.mark.asyncio
async def test_list_all_with_pagination(backend: SqliteVecBackend):
    """list_all supports offset and limit pagination."""
    ids = []
    for i in range(5):
        mid = await backend.add(
            user_id="test_user",
            content=f"Pagination test {i}",
            embedding=SAMPLE_EMBEDDING,
        )
        ids.append(mid)

    page1 = await backend.list_all("test_user", limit=2, offset=0)
    assert len(page1) == 2

    page2 = await backend.list_all("test_user", limit=2, offset=2)
    assert len(page2) == 2

    # Ensure no overlap between pages
    ids1 = {r["id"] for r in page1}
    ids2 = {r["id"] for r in page2}
    assert ids1.isdisjoint(ids2)


@pytest.mark.asyncio
async def test_delete_by_tag(backend: SqliteVecBackend):
    """delete_by_tag removes memories with matching tag."""
    mem_id = await backend.add(
        user_id="test_user",
        content="Tagged for deletion",
        embedding=SAMPLE_EMBEDDING,
        metadata={"tags": ["test-tag-to-delete"]},
    )

    count = await backend.delete_by_tag("test_user", "test-tag-to-delete")
    assert count >= 1

    # Should no longer appear in list
    results = await backend.list_all("test_user")
    assert mem_id not in [r["id"] for r in results]


@pytest.mark.asyncio
async def test_normalize_l2_zero_vector():
    """normalize_l2 returns unchanged vector for zero vector."""
    vec = [0.0, 0.0, 0.0]
    result = normalize_l2(vec)
    assert result == vec


@pytest.mark.asyncio
async def test_normalize_l2_nonzero():
    """normalize_l2 produces unit vector."""
    vec = [3.0, 4.0]
    result = normalize_l2(vec)
    import math
    magnitude = math.sqrt(sum(x ** 2 for x in result))
    assert abs(magnitude - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_delete_nonexistent(backend: SqliteVecBackend):
    """delete returns False for non-existent memory (covers line 365)."""
    success = await backend.delete("nonexistent-id")
    assert success is False


@pytest.mark.asyncio
async def test_update_nonexistent(backend: SqliteVecBackend):
    """update returns False for non-existent memory (covers line 390)."""
    success = await backend.update("nonexistent-id", "new content", {"key": "val"})
    assert success is False


@pytest.mark.asyncio
async def test_load_vec_extension(backend: SqliteVecBackend):
    """load_vec_extension loads sqlite-vec into a raw connection (covers lines 83-85)."""
    import sqlite3, sqlite_vec
    conn = sqlite3.connect(":memory:")
    load_vec_extension(conn)
    # Verify extension loaded by creating a vec0 table
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS test_vec USING vec0(embedding FLOAT[4])")
    conn.close()


@pytest.mark.asyncio
async def test_flag_memories_empty(backend: SqliteVecBackend):
    """_flag_memories returns early when ids list is empty (covers line 448-449)."""
    await backend._flag_memories([], "consolidated", True)
    # No crash = success
