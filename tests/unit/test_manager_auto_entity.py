"""Tests for manager.py _auto_create_entities and noise filter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memorymesh.memory.manager import MemoryManager
from memorymesh.memory.graph_store import GraphStore


SAMPLE_EMBEDDING = [0.1] * 384


@pytest.mark.asyncio
async def test_auto_create_entities_extracts_capitalized(memory_manager):
    """Test that _auto_create_entities extracts capitalized multi-word terms."""
    mock_graph = AsyncMock()
    mock_graph.create_entity = AsyncMock(return_value="entity-1")
    memory_manager.graph = mock_graph

    await memory_manager._auto_create_entities(
        "mem1", "We implemented the Knowledge Graph feature successfully", "test_user"
    )

    # Should have created entities for "Knowledge Graph" and "Graph" (if >= 3 chars)
    assert mock_graph.create_entity.call_count >= 1
    calls = [str(c) for c in mock_graph.create_entity.call_args_list]
    assert any("Knowledge Graph" in c for c in calls)


@pytest.mark.asyncio
async def test_auto_create_entities_extracts_pascal_case(memory_manager):
    """Test that _auto_create_entities extracts PascalCase names."""
    mock_graph = AsyncMock()
    mock_graph.create_entity = AsyncMock(return_value="entity-1")
    memory_manager.graph = mock_graph

    await memory_manager._auto_create_entities(
        "mem1", "Refactored MemoryManager to use GraphStore for entity storage", "test_user"
    )

    assert mock_graph.create_entity.call_count >= 2
    calls = [str(c) for c in mock_graph.create_entity.call_args_list]
    assert any("MemoryManager" in c for c in calls)
    assert any("GraphStore" in c for c in calls)


@pytest.mark.asyncio
async def test_auto_create_entities_skips_when_graph_none(memory_manager):
    """Test that _auto_create_entities returns silently when graph is None."""
    memory_manager.graph = None
    # Should not raise
    await memory_manager._auto_create_entities("mem1", "Some text", "test_user")


@pytest.mark.asyncio
async def test_auto_create_entities_skips_empty_text(memory_manager):
    """Test that _auto_create_entities returns silently when text is empty."""
    mock_graph = AsyncMock()
    memory_manager.graph = mock_graph

    await memory_manager._auto_create_entities("mem1", "", "test_user")
    mock_graph.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_auto_create_entities_handles_entity_creation_error(memory_manager):
    """Test that _auto_create_entities handles errors gracefully."""
    mock_graph = AsyncMock()
    mock_graph.create_entity = AsyncMock(side_effect=Exception("DB error"))
    memory_manager.graph = mock_graph

    # Should not raise
    await memory_manager._auto_create_entities(
        "mem1", "Knowledge Graph feature", "test_user"
    )


@pytest.mark.asyncio
async def test_auto_create_entities_skips_short_names(memory_manager):
    """Test that _auto_create_entities skips names shorter than 3 chars."""
    mock_graph = AsyncMock()
    mock_graph.create_entity = AsyncMock(return_value="entity-1")
    memory_manager.graph = mock_graph

    await memory_manager._auto_create_entities(
        "mem1", "I use AI for ML tasks", "test_user"
    )

    # AI and ML are too short, should not create entities
    mock_graph.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_noise_filter_penalizes_auto_save_tags(memory_manager):
    """Test that search_memory penalizes auto_save and auto_snapshot tags."""
    # Create memories with auto_save tags
    await memory_manager.backend.add(
        user_id="test_user", content="Auto snapshot entry",
        embedding=SAMPLE_EMBEDDING,
        metadata={"importance": 5, "level": "session", "tags": ["auto_save", "auto_snapshot"]},
        level="session",
    )
    # Create a normal memory
    await memory_manager.backend.add(
        user_id="test_user", content="Important user decision about architecture",
        embedding=SAMPLE_EMBEDDING,
        metadata={"importance": 5, "level": "user", "tags": ["decision"]},
        level="user",
    )

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        results = await memory_manager.search_memory(
            query="architecture decision", top_k=10, user_id="test_user"
        )

    # The auto_save memory should be penalized (lower score)
    # The user memory should rank higher
    assert len(results) >= 1
    # Find the scores (results are TypedDicts, use key access)
    auto_save_score = None
    user_score = None
    for r in results:
        tags = r["tags"] if isinstance(r, dict) else []
        if "auto_save" in tags:
            auto_save_score = r["score"]
        if "decision" in tags:
            user_score = r["score"]

    # If both found, user should have higher score
    if auto_save_score is not None and user_score is not None:
        assert user_score > auto_save_score
