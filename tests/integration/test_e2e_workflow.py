import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from memorymesh.mcp_server.handlers.base import ToolHandlers
from memorymesh.memory.session_store import SessionStore


SAMPLE_EMBEDDING = [0.1] * 384


@pytest_asyncio.fixture
async def session_store(session_config):
    store = SessionStore(session_config.db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_full_workflow(memory_manager, session_store):
    handlers = ToolHandlers(memory_manager, session_store)

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            r1 = await handlers.handle_remember({
                "content": "Tên tôi là Khang, thích phát triển AI với Python",
                "tags": ["giới-thiệu", "ai"],
                "importance": 4,
                "user_id": "e2e_user",
            })
            assert r1["status"] == "success"
            memory_id_1 = r1["data"]["id"]

            r2 = await handlers.handle_remember({
                "content": "Hà Nội là thủ đô của Việt Nam",
                "tags": ["địa-lý"],
                "importance": 3,
                "user_id": "e2e_user",
            })
            assert r2["status"] == "success"
            memory_id_2 = r2["data"]["id"]

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        recall = await handlers.handle_recall({
            "query": "tên và sở thích",
            "top_k": 10,
            "user_id": "e2e_user",
        })
    assert recall["status"] == "success"
    assert len(recall["data"]) >= 1
    contents = [m["content"] for m in recall["data"]]
    assert any("Khang" in c for c in contents)

    list_result = await handlers.handle_list_memories({
        "limit": 10,
        "offset": 0,
        "user_id": "e2e_user",
    })
    assert list_result["status"] == "success"
    assert len(list_result["data"]) == 2

    ping_result = await handlers.handle_ping({})
    assert ping_result["status"] == "success"
    assert ping_result["data"]["status"] == "ok"
    assert isinstance(ping_result["data"]["memory_count"], int)

    forget_result = await handlers.handle_forget({"memory_id": memory_id_1})
    assert forget_result["status"] == "success"
    assert forget_result["data"]["archived"] is True

    list_after = await handlers.handle_list_memories({
        "limit": 10,
        "offset": 0,
        "user_id": "e2e_user",
    })
    assert len(list_after["data"]) == 1
    assert list_after["data"][0]["id"] == memory_id_2


@pytest.mark.asyncio
async def test_multiple_users_isolation(memory_manager, session_store):
    handlers = ToolHandlers(memory_manager, session_store)

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            await handlers.handle_remember({
                "content": "Alice data",
                "user_id": "alice",
            })
            await handlers.handle_remember({
                "content": "Bob data",
                "user_id": "bob",
            })

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        alice_recall = await handlers.handle_recall({
            "query": "data",
            "top_k": 10,
            "user_id": "alice",
        })
        bob_recall = await handlers.handle_recall({
            "query": "data",
            "top_k": 10,
            "user_id": "bob",
        })

    alice_contents = [m["content"] for m in alice_recall["data"]]
    bob_contents = [m["content"] for m in bob_recall["data"]]
    assert "Alice data" in alice_contents
    assert "Bob data" in bob_contents
    assert "Alice data" not in bob_contents
    assert "Bob data" not in alice_contents
