import asyncio
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from memorymesh.mcp_server.handlers.base import ToolHandlers
from memorymesh.mcp_server.handlers.tracker import ConversationTracker
from memorymesh.mcp_server.handlers.semantic_filter import SemanticFilter
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
async def test_handle_recall_returns_context_restored(handlers):
    """recall response should include context_restored in meta field."""
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
            await handlers.handle_remember({
                "content": "Test context for signal",
                "user_id": "test_user",
            })

    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        result = await handlers.handle_recall({
            "query": "Test context for signal",
            "top_k": 5,
            "user_id": "test_user",
        })

    assert result["status"] == "success"
    assert "meta" in result
    assert "context_restored" in result["meta"]
    assert isinstance(result["meta"]["context_restored"], bool)
    assert "has_bootstrap" in result["meta"]


@pytest.mark.asyncio
async def test_handle_recall_context_restored_false_when_empty(handlers):
    """recall with no results should set context_restored to false."""
    with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        result = await handlers.handle_recall({
            "query": "nonexistent_xyzzy_12345",
            "top_k": 5,
            "user_id": "test_user",
        })

    assert result["status"] == "success"
    assert result["meta"]["context_restored"] is False
    assert result["meta"]["has_bootstrap"] is False


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
    assert result["data"]["message"] == "New session created"

    session = await session_store.get_session(result["data"]["session_id"])
    assert "Test prompt" in session["system_prompt"]
    assert "PERMANENT LOG DIRECTIVE" in session["system_prompt"]
    assert session["status"] == "active"

    session_id2 = result["data"]["session_id"]
    assert await handlers.get_current_session_id() == session_id2


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

    assert await handlers.get_current_session_id() == second_id


@pytest.mark.asyncio
async def test_handle_end_session(handlers, session_store):
    create = await handlers.handle_new_session({"user_id": "test_user"})
    session_id = create["data"]["session_id"]

    result = await handlers.handle_end_session({"session_id": session_id})
    assert result["status"] == "success"
    assert result["data"]["message"] == "Session ended"

    session = await session_store.get_session(session_id)
    assert session["status"] == "ended"
    assert await handlers.get_current_session_id() == ""


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

    snapshots = await session_store.get_workspace_snapshots(await handlers.get_current_session_id())
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


class TestConversationTrackerIntegration:
    @pytest.mark.asyncio
    async def test_record_tool_call_creates_tracker(self, handlers):
        await handlers.set_session("test_session_1")
        await handlers._record_tool_call("test_session_1", "remember", {"content": "test fact"})
        tracker = await handlers._get_tracker("test_session_1")
        assert isinstance(tracker, ConversationTracker)
        assert "[remember:test fact]" in tracker._tool_footprints

    @pytest.mark.asyncio
    async def test_record_and_on_milestone_commit(self, handlers):
        await handlers.set_session("test_session_2")
        await handlers._record_tool_call("test_session_2", "bash", {"command": "ls"})
        await handlers._record_tool_call("test_session_2", "write_file", {"content": "test"})
        tracker = await handlers._get_tracker("test_session_2")
        assert len(tracker._tool_footprints) == 2

        await handlers._on_milestone_commit("test_session_2")
        assert len(tracker._tool_footprints) == 0
        assert tracker._uncommitted_actions == 0

    @pytest.mark.asyncio
    async def test_tracker_isolation_by_session(self, handlers):
        await handlers._record_tool_call("session_a", "remember", {"content": "fact A"})
        await handlers._record_tool_call("session_b", "bash", {"command": "ls"})

        tracker_a = await handlers._get_tracker("session_a")
        tracker_b = await handlers._get_tracker("session_b")

        assert "[remember:fact A]" in tracker_a._tool_footprints
        assert "[bash]" in tracker_b._tool_footprints
        assert len(tracker_a._tool_footprints) == 1
        assert len(tracker_b._tool_footprints) == 1

    @pytest.mark.asyncio
    async def test_uncommitted_actions_tracking(self, handlers):
        await handlers.set_session("test_session_3")
        session_id = "test_session_3"

        await handlers._record_tool_call(session_id, "bash", {"command": "ls"})
        tracker = await handlers._get_tracker(session_id)
        assert tracker._uncommitted_actions == 1  # bash not in READ_ONLY_TOOLS

        tracker._uncommitted_actions = 0
        tracker._tool_footprints.append("[bash]")
        snapshot = tracker.flush()
        assert "[AUTO-SNAPSHOT]" in snapshot

        await handlers._on_milestone_commit(session_id)
        assert tracker._uncommitted_actions == 0
        assert tracker._has_unsaved_context is False

    @pytest.mark.asyncio
    async def test_note_tool_call_also_records_tracker(self, handlers):
        await handlers.set_session("test_session_4")
        sid = await handlers.get_current_session_id()
        await handlers._note_tool_call("remember", {"content": "via _note_tool_call"})
        tracker = await handlers._get_tracker(sid)
        assert "[remember:via _note_tool_call]" in tracker._tool_footprints


class TestSaveAutoToolContextUpgrade:
    SAMPLE_EMBEDDING = [0.1] * 384

    @pytest.mark.asyncio
    async def test_structured_extraction_remember(self, handlers):
        await handlers.set_session("test_auto_session")
        while not handlers._write_queue.empty():
            handlers._write_queue.get_nowait()

        with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
            with patch("memorymesh.memory.manager.MemoryManager._enrich_memory", new=AsyncMock()):
                await handlers.save_auto_tool_context(
                    "remember",
                    {"content": "Important decision: use sqlite-vec"},
                    {},
                )

        assert handlers._write_queue.qsize() == 1
        task = handlers._write_queue.get_nowait()
        assert "[DECISION]" in task["text"]
        assert task["importance"] == 4
        assert "remember" in task["tags"]

    @pytest.mark.asyncio
    async def test_structured_extraction_recall(self, handlers):
        await handlers.set_session("test_auto_session_2")
        await handlers.save_auto_tool_context(
            "recall",
            {"query": "project bootstrap context"},
            {"status": "success", "data": [{"id": "1"}, {"id": "2"}, {"id": "3"}]},
        )

        assert handlers._write_queue.qsize() >= 1
        task = None
        while not handlers._write_queue.empty():
            task = handlers._write_queue.get_nowait()
        assert task is not None
        assert "[RECALL]" in task["text"]
        assert "project bootstrap" in task["text"]
        assert "3 results" in task["text"]

    @pytest.mark.asyncio
    async def test_noise_filtered_out(self, handlers):
        await handlers.set_session("test_noise_session")
        await handlers.save_auto_tool_context(
            "bash",
            {"command": "echo hello"},
            {},
        )
        # "echo hello" is short and trivial - should be filtered
        assert handlers._write_queue.qsize() == 0


class TestSemanticFilterIntegration:
    @pytest.mark.asyncio
    async def test_filter_applied_in_save_auto_context(self, handlers):
        await handlers.set_session("test_filter_session")
        await handlers.save_auto_tool_context(
            "remember",
            {"content": "hi"},
            {},
        )
        assert handlers._write_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_meaningful_content_passes_filter(self, handlers):
        await handlers.set_session("test_filter_session_2")
        await handlers.save_auto_tool_context(
            "remember",
            {"content": "Refactored the search fallback to use hybrid backend"},
            {},
        )
        assert handlers._write_queue.qsize() >= 1
        while not handlers._write_queue.empty():
            handlers._write_queue.get_nowait()

