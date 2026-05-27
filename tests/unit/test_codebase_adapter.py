import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from memorymesh.memory.codebase_adapter import CodebaseDBAdapter

@pytest.fixture
def adapter():
    return CodebaseDBAdapter(db_path="dummy.db")

@pytest.mark.asyncio
async def test_init_no_path():
    adapter = CodebaseDBAdapter(db_path=None)
    await adapter.initialize()
    assert adapter.is_available is False

@pytest.mark.asyncio
async def test_init_success(adapter):
    mock_db = AsyncMock()
    with patch('aiosqlite.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_db
        await adapter.initialize()
        mock_connect.assert_called_once()
        assert adapter.is_available is True
        mock_db.execute.assert_any_call("PRAGMA query_only=ON")

@pytest.mark.asyncio
async def test_init_timeout(adapter):
    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        await adapter.initialize()
        assert adapter.is_available is False

@pytest.mark.asyncio
async def test_init_error(adapter):
    with patch('aiosqlite.connect', side_effect=Exception("DB Error")):
        await adapter.initialize()
        assert adapter.is_available is False

@pytest.mark.asyncio
async def test_close(adapter):
    mock_db = AsyncMock()
    adapter._db = mock_db
    adapter._available = True
    await adapter.close()
    mock_db.close.assert_called_once()
    assert adapter.is_available is False
    assert adapter._db is None

@pytest.mark.asyncio
async def test_search_entities_not_available(adapter):
    assert await adapter.search_entities("test") == []

async def mock_wait_for(coro, timeout):
    return await coro

@pytest.mark.asyncio
async def test_search_entities_success(adapter):
    mock_db = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchall.return_value = [
        {"id": "1", "name": "Test Concept", "type": "concept", "properties": '{"summary": "hello"}'}
    ]
    mock_db.execute.return_value = mock_cursor
    adapter._db = mock_db
    adapter._available = True

    # Patch wait_for to just return the coroutine result
    with patch('asyncio.wait_for', new_callable=AsyncMock) as m_wait_for:
        m_wait_for.side_effect = mock_wait_for
        results = await adapter.search_entities("test")
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["properties"] == {"summary": "hello"}

@pytest.mark.asyncio
async def test_search_entities_timeout(adapter):
    adapter._db = AsyncMock()
    adapter._available = True
    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        assert await adapter.search_entities("test") == []

@pytest.mark.asyncio
async def test_search_entities_error(adapter):
    adapter._db = AsyncMock()
    adapter._available = True
    with patch('asyncio.wait_for', side_effect=Exception("query failed")):
        assert await adapter.search_entities("test") == []

@pytest.mark.asyncio
async def test_search_relations_not_available(adapter):
    assert await adapter.search_relations("1") == []

@pytest.mark.asyncio
async def test_search_relations_success(adapter):
    mock_db = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchall.return_value = [
        {"id": "r1", "source_id": "1", "target_id": "2", "relation_type": "depends_on", "weight": 1.0, "source_name": "A", "target_name": "B"}
    ]
    mock_db.execute.return_value = mock_cursor
    adapter._db = mock_db
    adapter._available = True

    with patch('asyncio.wait_for', new_callable=AsyncMock) as m_wait_for:
        m_wait_for.side_effect = mock_wait_for
        results = await adapter.search_relations("1")
        assert len(results) == 1
        assert results[0]["id"] == "r1"

@pytest.mark.asyncio
async def test_search_relations_timeout(adapter):
    adapter._db = AsyncMock()
    adapter._available = True
    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        assert await adapter.search_relations("1") == []

@pytest.mark.asyncio
async def test_search_relations_error(adapter):
    adapter._db = AsyncMock()
    adapter._available = True
    with patch('asyncio.wait_for', side_effect=Exception("query error")):
        assert await adapter.search_relations("1") == []