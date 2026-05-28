"""Extended tests for consolidation.py — fact consolidation, memory decay, edge cases."""

import pytest
from unittest.mock import AsyncMock, patch

from memorymesh.memory.consolidation import ConsolidationEngine


SAMPLE_EMBEDDING = [0.1] * 384


@pytest.fixture
def consolidation(app_config):
    engine = ConsolidationEngine(app_config, None, None)
    engine.threshold = 0.6
    return engine


class TestFactConsolidation:
    @pytest.mark.asyncio
    async def test_disabled_returns_zero(self, consolidation):
        consolidation.config.consolidation.enabled = False
        result = await consolidation.run_fact_consolidation("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_no_facts_returns_zero(self, consolidation):
        consolidation.config.consolidation.enabled = True
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = []
        consolidation.backend = mock_backend

        result = await consolidation.run_fact_consolidation("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_single_fact_returns_zero(self, consolidation):
        consolidation.config.consolidation.enabled = True
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "f1", "metadata": {"tags": ["atomic_fact"]}},
        ]
        consolidation.backend = mock_backend

        result = await consolidation.run_fact_consolidation("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_fact_resolve_no_contradictions(self, consolidation):
        consolidation.config.consolidation.enabled = True
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "f1", "content": "Alice likes cats", "metadata": {"tags": ["atomic_fact"]}, "embedding": [1.0, 0.0]},
            {"id": "f2", "content": "Bob likes dogs", "metadata": {"tags": ["atomic_fact"]}, "embedding": [0.0, 1.0]},
        ]
        mock_backend.add.return_value = "mock_id"
        mock_router = AsyncMock()
        mock_router.call_llm.return_value = '{"contradictions_found": false, "resolutions": []}'

        consolidation.backend = mock_backend
        consolidation.router = mock_router

        result = await consolidation.run_fact_consolidation("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_fact_resolve_with_contradictions(self, consolidation):
        consolidation.config.consolidation.enabled = True
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "f1", "content": "Server IP is 10.0.0.1", "metadata": {"tags": ["atomic_fact"]}, "embedding": [1.0, 0.0]},
            {"id": "f2", "content": "Server IP is 10.0.0.2", "metadata": {"tags": ["atomic_fact"]}, "embedding": [1.0, 0.0]},
        ]
        mock_backend.add.return_value = "mock_id"
        mock_router = AsyncMock()
        mock_router.call_llm.return_value = '{"contradictions_found": true, "resolutions": [{"keep": "Server IP is 10.0.0.1", "remove": ["Server IP is 10.0.0.2"], "reason": "Newer entry overrides older"}]}'

        consolidation.backend = mock_backend
        consolidation.router = mock_router

        result = await consolidation.run_fact_consolidation("test_user")
        assert result >= 1  # At least one group resolved


class TestResolveFactContradictions:
    @pytest.mark.asyncio
    async def test_skip_single_fact_group(self, consolidation):
        group = [{"id": "f1", "content": "test", "metadata": {}}]
        mock_router = AsyncMock()
        consolidation.router = mock_router

        await consolidation._resolve_fact_contradictions(group, "test_user")
        mock_router.call_llm.assert_not_called()  # Skip because len < 2


class TestMemoryDecay:
    @pytest.mark.asyncio
    async def test_ttl_zero_returns_zero(self, consolidation):
        consolidation.config.consolidation.session_memory_ttl_days = 0
        mock_backend = AsyncMock()
        consolidation.backend = mock_backend

        result = await consolidation.run_memory_decay("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_non_session_level(self, consolidation):
        consolidation.config.consolidation.session_memory_ttl_days = 7
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "m1", "content": "test", "metadata": {"level": "user", "importance": 2}},
        ]
        consolidation.backend = mock_backend

        result = await consolidation.run_memory_decay("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_consolidated_memories(self, consolidation):
        consolidation.config.consolidation.session_memory_ttl_days = 7
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "m1", "content": "test", "metadata": {"level": "session", "importance": 2, "consolidated": True}},
        ]
        consolidation.backend = mock_backend

        result = await consolidation.run_memory_decay("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_workspace_tagged(self, consolidation):
        consolidation.config.consolidation.session_memory_ttl_days = 7
        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "m1", "content": "test", "metadata": {"level": "session", "importance": 2, "workspace_path": "/project"}},
        ]
        consolidation.backend = mock_backend

        result = await consolidation.run_memory_decay("test_user")
        assert result == 0
