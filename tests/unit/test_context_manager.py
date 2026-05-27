import pytest
from unittest.mock import MagicMock
from memorymesh.memory.context_manager import ContextManager, Cursor, _build_dynamic_score_query

def test_cursor_from_to_dict():
    c = Cursor.from_dict({"last_score": 10.0, "last_id": "test1", "page": 2})
    assert c.last_score == 10.0
    assert c.last_id == "test1"
    assert c.page == 2
    
    assert c.to_dict() == {"last_score": 10.0, "last_id": "test1", "page": 2}

    c_empty = Cursor.from_dict(None)
    assert c_empty.last_score == 9999.0
    assert c_empty.last_id == ""
    assert c_empty.page == 1

def test_build_dynamic_score_query():
    cur = Cursor()
    sql, params = _build_dynamic_score_query("user1", cur)
    assert "FROM memories" in sql
    assert params == [2.0, 1.5, 1.0, 24.0, "user1", 9999.0, 9999.0, "", 50]
    
    sql2, params2 = _build_dynamic_score_query("user2", cur, level_weights={"session": 3.0, "user": 2.0, "knowledge": 1.5})
    assert params2 == [3.0, 2.0, 1.5, 24.0, "user2", 9999.0, 9999.0, "", 50]

class MockAsyncCursor:
    def __init__(self, rows):
        self.rows = rows
        self._idx = 0
    
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self._idx < len(self.rows):
            row = self.rows[self._idx]
            self._idx += 1
            return row
        raise StopAsyncIteration

class MockAsyncContextManager:
    def __init__(self, rows):
        self.cursor = MockAsyncCursor(rows)
    
    async def __aenter__(self):
        return self.cursor
        
    async def __aexit__(self, exc_type, exc, tb):
        pass

class MockDB:
    def __init__(self, rows):
        self.rows = rows
        
    def execute(self, sql, params):
        return MockAsyncContextManager(self.rows)

class MockBackend:
    def __init__(self, rows):
        self._db = MockDB(rows)

@pytest.mark.asyncio
async def test_get_context_page_success():
    rows = [
        {
            "id": "mem1", "content": "hello world", "metadata_json": '{"tag": "test"}',
            "importance": 4, "level": "session", "created_at": "2023-01-01T00:00:00Z",
            "dynamic_score": 8.0
        },
        {
            "id": "mem2", "content": "foo bar", "metadata_json": "{}",
            "importance": 2, "level": "user", "created_at": "2023-01-01T00:00:00Z",
            "dynamic_score": 4.0
        }
    ]
    backend = MockBackend(rows)
    manager = ContextManager(backend)
    
    res = await manager.get_context_page("user1", max_tokens=100)
    assert len(res["results"]) == 2
    assert res["results"][0]["id"] == "mem1"
    assert res["results"][0]["score"] == 8.0
    assert res["has_more"] is False
    assert res["next_cursor"] is None

@pytest.mark.asyncio
async def test_get_context_page_tuple_rows():
    rows = [
        ("mem1", "hello", "{}", 3, "user", "now", 5.0)
    ]
    backend = MockBackend(rows)
    manager = ContextManager(backend)
    
    res = await manager.get_context_page("user1", max_tokens=100)
    assert len(res["results"]) == 1
    assert res["results"][0]["id"] == "mem1"

@pytest.mark.asyncio
async def test_get_context_page_token_limit():
    rows = [
        {
            "id": "mem1", "content": "a " * 500, "metadata_json": '{}',
            "importance": 4, "level": "session", "created_at": "2023-01-01T00:00:00Z",
            "dynamic_score": 8.0
        },
        {
            "id": "mem2", "content": "b " * 500, "metadata_json": '{}',
            "importance": 4, "level": "session", "created_at": "2023-01-01T00:00:00Z",
            "dynamic_score": 7.0
        }
    ]
    backend = MockBackend(rows)
    manager = ContextManager(backend)
    
    res = await manager.get_context_page("user1", max_tokens=400)
    assert len(res["results"]) == 1
    assert res["results"][0]["id"] == "mem1"
    assert res["has_more"] is True
    assert res["next_cursor"] is not None
    assert res["next_cursor"]["last_score"] == 8.0
    assert res["next_cursor"]["last_id"] == "mem1"

@pytest.mark.asyncio
async def test_get_context_page_no_db():
    backend = MagicMock()
    backend._db = None
    manager = ContextManager(backend)
    res = await manager.get_context_page("user1")
    assert res["results"] == []

@pytest.mark.asyncio
async def test_get_context_page_db_error():
    class ErrorDB:
        def execute(self, sql, params):
            raise Exception("DB Error")
            
    backend = MagicMock()
    backend._db = ErrorDB()
    manager = ContextManager(backend)
    res = await manager.get_context_page("user1")
    assert res["results"] == []

@pytest.mark.asyncio
async def test_get_context_page_empty_result():
    backend = MockBackend([])
    manager = ContextManager(backend)
    res = await manager.get_context_page("user1")
    assert res["results"] == []