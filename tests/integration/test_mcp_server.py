import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from memorymesh.mcp_server.handlers import ToolHandlers
from memorymesh.memory.session_store import SessionStore
from memorymesh.config import AppConfig


SAMPLE_EMBEDDING = [0.1] * 384


@pytest_asyncio.fixture
async def session_store(session_config):
    store = SessionStore(session_config.db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def handlers(memory_manager, session_store):
    return ToolHandlers(memory_manager, session_store)


@pytest.mark.asyncio
async def test_handle_ping(handlers):
    result = await handlers.handle_ping({})
    assert result["status"] == "success"
    assert result["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_handle_remember(handlers):
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            result = await handlers.handle_remember({
                "content": "Integration test memory",
                "tags": ["test"],
                "importance": 3,
                "user_id": "test_user",
            })
    assert result["status"] == "success"
    assert "id" in result["data"]


@pytest.mark.asyncio
async def test_handle_remember_with_error(memory_manager, handlers):
    from memorymesh.errors import ValidationError
    with patch.object(memory_manager, "add_memory", new=AsyncMock(side_effect=ValidationError("Invalid input"))):
        result = await handlers.handle_remember({
            "content": "test",
        })
        assert result["status"] == "error"
        assert "Invalid input" in result["error"]


@pytest.mark.asyncio
async def test_handle_recall(handlers):
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            await handlers.handle_remember({
                "content": "Hà Nội là thủ đô",
                "user_id": "test_user",
            })

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        result = await handlers.handle_recall({
            "query": "thủ đô",
            "top_k": 5,
            "user_id": "test_user",
        })
    assert result["status"] == "success"
    assert len(result["data"]) >= 1


@pytest.mark.asyncio
async def test_handle_forget(handlers):
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            mem_result = await handlers.handle_remember({
                "content": "To forget",
                "user_id": "test_user",
            })

    memory_id = mem_result["data"]["id"]
    result = await handlers.handle_forget({"memory_id": memory_id})
    assert result["status"] == "success"
    assert result["data"]["archived"] is True


@pytest.mark.asyncio
async def test_handle_list_memories(handlers):
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            await handlers.handle_remember({
                "content": "List test",
                "user_id": "test_user",
            })

    result = await handlers.handle_list_memories({
        "limit": 10,
        "offset": 0,
        "user_id": "test_user",
    })
    assert result["status"] == "success"
    assert len(result["data"]) >= 1


@pytest.mark.asyncio
async def test_handle_new_session(handlers, session_store):
    result = await handlers.handle_new_session({
        "system_prompt": "Test prompt",
        "user_id": "test_user",
    })
    assert result["status"] == "success"
    assert "session_id" in result["data"]
    assert result["data"]["message"] == "Session mới đã được tạo"

    session = await session_store.get_session(result["data"]["session_id"])
    assert session["system_prompt"] == "Test prompt"
    assert session["status"] == "active"

    session_id2 = result["data"]["session_id"]
    assert handlers._current_session_id == session_id2


@pytest.mark.asyncio
async def test_handle_new_session_auto_ends_previous(handlers, session_store):
    first = await handlers.handle_new_session({"user_id": "test_user"})
    first_id = first["data"]["session_id"]

    second = await handlers.handle_new_session({"user_id": "test_user"})
    second_id = second["data"]["session_id"]

    first_session = await session_store.get_session(first_id)
    assert first_session["status"] == "ended"

    second_session = await session_store.get_session(second_id)
    assert second_session["status"] == "active"

    assert handlers._current_session_id == second_id


@pytest.mark.asyncio
async def test_handle_end_session(handlers, session_store):
    create = await handlers.handle_new_session({"user_id": "test_user"})
    session_id = create["data"]["session_id"]

    result = await handlers.handle_end_session({"session_id": session_id})
    assert result["status"] == "success"
    assert result["data"]["message"] == "Session đã kết thúc"

    session = await session_store.get_session(session_id)
    assert session["status"] == "ended"
    assert handlers._current_session_id == ""


@pytest.mark.asyncio
async def test_handle_end_session_default_current(handlers, session_store):
    await handlers.handle_new_session({"user_id": "test_user"})

    result = await handlers.handle_end_session({})
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_handle_save_workspace_context(handlers, session_store):
    await handlers.handle_new_session({"user_id": "test_user"})

    result = await handlers.handle_save_workspace_context({
        "workspace_path": ".",
        "user_id": "test_user",
    })
    assert result["status"] == "success"
    assert "memory_id" in result["data"]
    assert "snapshot" in result["data"]
    assert "files" in result["data"]["snapshot"]
    assert "dependencies" in result["data"]["snapshot"]

    snapshots = await session_store.get_workspace_snapshots(handlers._current_session_id)
    assert len(snapshots) >= 1


@pytest.mark.asyncio
async def test_handle_resume_session(handlers, session_store):
    first = await handlers.handle_new_session({
        "system_prompt": "Initial prompt",
        "user_id": "test_user",
    })
    first_id = first["data"]["session_id"]

    await session_store.log_context(first_id, "user", "Hello")
    await session_store.log_context(first_id, "assistant", "Hi there")

    second = await handlers.handle_new_session({"user_id": "test_user"})

    result = await handlers.handle_resume_session({
        "session_id": first_id,
        "top_k": 5,
        "user_id": "test_user",
    })
    assert result["status"] == "success"
    assert result["data"]["session"]["session_id"] == first_id
    assert len(result["data"]["context_log"]) >= 2
    assert "message" in result["data"]
