import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from memorymesh.mcp_server.handlers.base import ToolHandlers


@pytest.fixture
def mock_mgr():
    m = MagicMock()
    m.config.default_user_id = "user1"
    m.backend.list_by_tag = AsyncMock(return_value=[])
    return m


@pytest.fixture
def mock_store():
    s = AsyncMock()
    s.get_context_log = AsyncMock(return_value=[])
    s.get_workspace_snapshots = AsyncMock(return_value=[])
    return s


class TestContextDelta:
    async def test_empty_session_returns_empty(self, mock_mgr, mock_store):
        handlers = ToolHandlers(mock_mgr, mock_store)
        handlers._global_bootstrap_ram_cache = {}
        result = await handlers._get_context_delta("session123", "user1", "", 800)
        assert result == ""

    async def test_context_delta_under_token_budget(self, mock_mgr, mock_store):
        mock_store.get_context_log = AsyncMock(return_value=[
            {"role": "user", "content": "test message"},
            {"role": "assistant", "content": "response"},
        ])
        handlers = ToolHandlers(mock_mgr, mock_store)
        handlers._global_bootstrap_ram_cache = {}
        result = await handlers._get_context_delta("session123", "user1", "", 800)
        assert len(result) > 0
        assert "Recent Activity" in result

    async def test_context_delta_truncates_to_token_budget(self, mock_mgr, mock_store):
        mock_store.get_context_log = AsyncMock(return_value=[
            {"role": "user", "content": "x" * 5000},
        ])
        handlers = ToolHandlers(mock_mgr, mock_store)
        handlers._global_bootstrap_ram_cache = {}
        result = await handlers._get_context_delta("session123", "user1", "", 10)
        assert "[truncated]" in result or len(result) < 500

    async def test_context_delta_includes_milestone_from_cache(self, mock_mgr, mock_store):
        handlers = ToolHandlers(mock_mgr, mock_store)
        handlers._global_bootstrap_ram_cache = {"user1": "Previous session summary"}
        result = await handlers._get_context_delta("session123", "user1", "", 800)
        assert "Previous Session" in result
