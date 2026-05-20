import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from memorymesh.memory.manager import MemoryManager
from memorymesh.errors import ValidationError


SAMPLE_EMBEDDING = [0.1] * 384


@pytest.fixture(autouse=True)
def mock_embedding():
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        yield


@pytest.fixture(autouse=True)
def mock_router_call():
    with patch.object(MemoryManager, "_enrich_memory", new=AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_add_memory_success(memory_manager):
    memory_id = await memory_manager.add_memory(
        text="Test memory content",
        tags=["test"],
        importance=3,
        user_id="test_user",
    )
    assert memory_id is not None
    assert isinstance(memory_id, str)


@pytest.mark.asyncio
async def test_add_memory_exceeds_max_length(memory_manager):
    long_text = "x" * (memory_manager.config.max_memory_length + 1)
    with pytest.raises(ValidationError, match="max length"):
        await memory_manager.add_memory(text=long_text)


@pytest.mark.asyncio
async def test_add_memory_invalid_importance_below(memory_manager):
    with pytest.raises(ValidationError, match="Importance"):
        await memory_manager.add_memory(text="test", importance=0)


@pytest.mark.asyncio
async def test_add_memory_invalid_importance_above(memory_manager):
    with pytest.raises(ValidationError, match="Importance"):
        await memory_manager.add_memory(text="test", importance=6)


@pytest.mark.asyncio
async def test_add_memory_default_user_id(memory_manager):
    memory_id = await memory_manager.add_memory(text="test")
    assert memory_id is not None


@pytest.mark.asyncio
async def test_search_memory(memory_manager):
    await memory_manager.add_memory(
        text="Hà Nội là thủ đô",
        tags=["địa lý"],
        importance=4,
        user_id="test_user",
    )

    results = await memory_manager.search_memory(
        query="thủ đô",
        top_k=5,
        user_id="test_user",
    )
    assert len(results) >= 1
    assert results[0]["content"] == "Hà Nội là thủ đô"
    assert results[0]["tags"] == ["địa lý"]
    assert results[0]["importance"] == 4


@pytest.mark.asyncio
async def test_search_memory_empty_result(memory_manager):
    results = await memory_manager.search_memory(
        query="nothing",
        top_k=5,
        user_id="nonexistent",
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_forget_memory(memory_manager):
    memory_id = await memory_manager.add_memory(
        text="To be forgotten",
        user_id="test_user",
    )
    success = await memory_manager.forget_memory(memory_id)
    assert success is True

    results = await memory_manager.search_memory(
        query="forgotten",
        top_k=5,
        user_id="test_user",
    )
    ids = [r["id"] for r in results]
    assert memory_id not in ids


@pytest.mark.asyncio
async def test_forget_memory_nonexistent(memory_manager):
    # ChromaDB silently ignores non-existent IDs
    success = await memory_manager.forget_memory("non-existent-id")
    assert success is True


@pytest.mark.asyncio
async def test_list_memories(memory_manager):
    ids = []
    for i in range(3):
        mid = await memory_manager.add_memory(
            text=f"Memory {i}",
            user_id="test_user",
        )
        ids.append(mid)

    all_memories = await memory_manager.list_memories(
        limit=10, offset=0, user_id="test_user",
    )
    assert len(all_memories) == 3
    assert all(m["user_id"] == "test_user" for m in all_memories)


@pytest.mark.asyncio
async def test_list_memories_default_user(memory_manager):
    mid = await memory_manager.add_memory(text="Default user test")
    results = await memory_manager.list_memories()
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_search_memory_token_budget(memory_manager):
    long_content = "word " * 300
    await memory_manager.add_memory(
        text=long_content,
        user_id="test_user",
    )

    results = await memory_manager.search_memory(
        query="word",
        top_k=5,
        user_id="test_user",
    )
    assert len(results) >= 1
