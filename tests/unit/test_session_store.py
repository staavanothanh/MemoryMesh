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
    assert session["status"] == "active"


@pytest.mark.asyncio
async def test_end_session(store):
    session_id = await store.create_session("test_user")
    await store.end_session(session_id)

    session = await store.get_session(session_id)
    assert session["status"] == "ended"
    assert session["ended_at"] is not None


@pytest.mark.asyncio
async def test_auto_close_stale(store):
    s1 = await store.create_session("test_user", auto_close_stale=False)
    s2 = await store.create_session("test_user", auto_close_stale=True)

    s1_data = await store.get_session(s1)
    assert s1_data["status"] == "ended"

    s2_data = await store.get_session(s2)
    assert s2_data["status"] == "active"


@pytest.mark.asyncio
async def test_get_active_session(store):
    s1 = await store.create_session("user_a", auto_close_stale=False)
    active = await store.get_active_session("user_a")
    assert active["session_id"] == s1

    await store.end_session(s1)
    active = await store.get_active_session("user_a")
    assert active is None


@pytest.mark.asyncio
async def test_list_sessions(store):
    for i in range(3):
        await store.create_session(f"user_{i}")

    sessions = await store.list_sessions("user_0")
    assert len(sessions) == 1
    assert sessions[0]["user_id"] == "user_0"


@pytest.mark.asyncio
async def test_list_sessions_by_status(store):
    s1 = await store.create_session("test_user", auto_close_stale=False)
    await store.end_session(s1)
    await store.create_session("test_user", auto_close_stale=False)

    active = await store.list_sessions("test_user", status="active")
    assert len(active) == 1
    assert active[0]["session_id"] != s1

    ended = await store.list_sessions("test_user", status="ended")
    assert len(ended) == 1
    assert ended[0]["session_id"] == s1


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


@pytest.mark.asyncio
async def test_workspace_snapshot(store):
    session_id = await store.create_session("test_user")
    snapshot = {"files": ["a.py", "b.py"], "git": {"commits": ["abc"]}}
    await store.save_workspace_snapshot(session_id, snapshot)

    snapshots = await store.get_workspace_snapshots(session_id)
    assert len(snapshots) == 1
    assert snapshots[0]["snapshot_data"]["files"] == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_create_session_no_auto_close(store):
    s1 = await store.create_session("test_user", auto_close_stale=False)
    s2 = await store.create_session("test_user", auto_close_stale=False)

    s1_data = await store.get_session(s1)
    s2_data = await store.get_session(s2)
    assert s1_data["status"] == "active"
    assert s2_data["status"] == "active"
