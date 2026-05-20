import pytest
import pytest_asyncio
from memorymesh.memory.session_store import SessionStore


@pytest_asyncio.fixture
async def store():
    s = SessionStore(":memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_and_get_session(store):
    session_id = await store.create_session("test_user", "system prompt", "/workspace")
    assert session_id is not None

    session = await store.get_session(session_id)
    assert session["user_id"] == "test_user"
    assert session["system_prompt"] == "system prompt"
    assert session["workspace_path"] == "/workspace"


@pytest.mark.asyncio
async def test_list_sessions(store):
    ids = []
    for i in range(3):
        sid = await store.create_session(f"user_{i}")
        ids.append(sid)

    sessions = await store.list_sessions("user_0")
    assert len(sessions) == 1
    assert sessions[0]["user_id"] == "user_0"


@pytest.mark.asyncio
async def test_update_system_prompt(store):
    session_id = await store.create_session("test_user")
    await store.update_system_prompt(session_id, "new prompt")

    session = await store.get_session(session_id)
    assert session["system_prompt"] == "new prompt"


@pytest.mark.asyncio
async def test_log_and_get_context(store):
    session_id = await store.create_session("test_user")
    await store.log_context(session_id, "user", "Hello")
    await store.log_context(session_id, "assistant", "Hi there", "remember", '{"content":"test"}')

    log = await store.get_context_log(session_id)
    assert len(log) == 2
    assert log[0]["role"] == "user"
    assert log[0]["content"] == "Hello"
    assert log[1]["role"] == "assistant"
    assert log[1]["tool_name"] == "remember"


@pytest.mark.asyncio
async def test_get_context_limit(store):
    session_id = await store.create_session("test_user")
    for i in range(10):
        await store.log_context(session_id, "user", f"msg {i}")

    log = await store.get_context_log(session_id, limit=3)
    assert len(log) == 3


@pytest.mark.asyncio
async def test_get_nonexistent_session(store):
    session = await store.get_session("nonexistent")
    assert session is None
