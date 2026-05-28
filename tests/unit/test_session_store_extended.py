"""Extended tests for session_store.py — mark_deleted, hard_delete, raw_log, edge cases."""

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
async def test_mark_deleted(store):
    session_id = await store.create_session("test_user")
    await store.mark_deleted(session_id)
    session = await store.get_session(session_id)
    assert session["status"] == "ended"
    assert session["deleted"] == 1


@pytest.mark.asyncio
async def test_hard_delete_removes_session_and_relations(store):
    session_id = await store.create_session("test_user")
    await store.log_context(session_id, "user", "Hello")
    await store.save_workspace_snapshot(session_id, {"files": ["a.py"]})

    await store.hard_delete_session(session_id)
    session = await store.get_session(session_id)
    assert session is None

    log = await store.get_context_log(session_id)
    assert len(log) == 0


@pytest.mark.asyncio
async def test_get_context_log_count(store):
    session_id = await store.create_session("test_user")
    await store.log_context(session_id, "user", "msg 1")
    await store.log_context(session_id, "user", "msg 2")
    count = await store.get_context_log_count(session_id)
    assert count == 2


@pytest.mark.asyncio
async def test_get_context_log_count_empty(store):
    session_id = await store.create_session("test_user")
    count = await store.get_context_log_count(session_id)
    assert count == 0


@pytest.mark.asyncio
async def test_create_session_with_empty_user_id(store):
    with pytest.MonkeyPatch.context() as mp:
        from memorymesh.config import AppConfig
        mp.setattr(AppConfig, "from_env", staticmethod(lambda: AppConfig(
            default_user_id="fallback_user",
            router=None, sqlite_vec=None, session=None,
            consolidation=None,
        )))
        session_id = await store.create_session("")
        session = await store.get_session(session_id)
        assert session["user_id"] == "fallback_user"


@pytest.mark.asyncio
async def test_list_sessions_with_deleted(store):
    s1 = await store.create_session("test_user", auto_close_stale=False)
    await store.mark_deleted(s1)
    s2 = await store.create_session("test_user", auto_close_stale=False)

    sessions = await store.list_sessions("test_user", include_deleted=False)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == s2

    sessions_all = await store.list_sessions("test_user", include_deleted=True)
    assert len(sessions_all) == 2


@pytest.mark.asyncio
async def test_list_sessions_by_status(store):
    s1 = await store.create_session("test_user", auto_close_stale=False)
    await store.end_session(s1)
    s2 = await store.create_session("test_user", auto_close_stale=False)

    active = await store.list_sessions("test_user", status="active")
    assert len(active) == 1
    assert active[0]["session_id"] == s2


@pytest.mark.asyncio
async def test_get_active_session_with_multiple(store):
    """get_active_session should handle situations with no active sessions."""
    # With no sessions, should return None
    active = await store.get_active_session("nonexistent_user")
    assert active is None


@pytest.mark.asyncio
async def test_raw_log_basic(store):
    session_id = await store.create_session("test_user")
    from datetime import datetime, timezone
    import json
    now = datetime.now(timezone.utc).isoformat()
    # Directly insert a raw log entry
    await store._db.execute(
        "INSERT INTO raw_log (session_id, tool_name, input_payload, output_payload, is_compressed, execution_time_ms, status) VALUES (?, ?, ?, ?, 1, ?, ?)",
        (session_id, "remember", b"test_input", b"test_output", 50.0, "success"),
    )
    await store._db.commit()

    log = await store.get_raw_log(session_id)
    assert len(log) >= 1
    assert log[0]["tool_name"] == "remember"
    assert log[0]["status"] == "success"


@pytest.mark.asyncio
async def test_raw_log_detail_returns_none_for_missing(store):
    detail = await store.get_raw_log_detail(99999)
    assert detail is None


@pytest.mark.asyncio
async def test_search_raw_log_by_tool(store):
    session_id = await store.create_session("test_user")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    await store._db.execute(
        "INSERT INTO raw_log (session_id, tool_name, input_payload, output_payload, is_compressed, execution_time_ms, status) VALUES (?, ?, ?, ?, 1, ?, ?)",
        (session_id, "bash", b"ls", b"files", 10.0, "success"),
    )
    await store._db.execute(
        "INSERT INTO raw_log (session_id, tool_name, input_payload, output_payload, is_compressed, execution_time_ms, status) VALUES (?, ?, ?, ?, 1, ?, ?)",
        (session_id, "recall", b"test", b"results", 20.0, "success"),
    )
    await store._db.commit()

    bash_logs = await store.search_raw_log(tool_name="bash", limit=10)
    assert len(bash_logs) >= 1
    recall_logs = await store.search_raw_log(tool_name="recall")
    assert len(recall_logs) >= 1
    recall_logs = await store.search_raw_log(status="success")
    assert len(recall_logs) >= 1


@pytest.mark.asyncio
async def test_get_workspace_snapshots_empty(store):
    snapshots = await store.get_workspace_snapshots("nonexistent")
    assert len(snapshots) == 0
