import pytest
import pytest_asyncio
import aiosqlite

from memorymesh.memory.instinct_store import InstinctStore
from memorymesh.memory.instinct_manager import InstinctManager
from memorymesh.utils.tool_middleware import ToolExecutionMiddleware


@pytest_asyncio.fixture
async def middleware():
    store = InstinctStore(":memory:")
    store._db = await aiosqlite.connect(":memory:")
    store._db.row_factory = aiosqlite.Row
    await store._db.execute("""
        CREATE TABLE IF NOT EXISTS instincts_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            trigger_regex TEXT NOT NULL,
            reaction TEXT NOT NULL,
            confidence_score REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            active INTEGER NOT NULL DEFAULT 1
        )
    """)
    await store._db.execute("""
        CREATE INDEX IF NOT EXISTS idx_instincts_v2_project
        ON instincts_v2(project_id, confidence_score DESC)
    """)
    await store._db.commit()
    manager = InstinctManager(store)
    await manager.load_all()
    mw = ToolExecutionMiddleware(manager)
    mw.set_project("test_project")
    yield (mw, store, manager)
    await store._db.close()


class TestToolExecutionMiddleware:
    async def test_record_call_stores_action(self, middleware):
        mw, _, _ = middleware
        mw.record_call("edit", {"file": "test.py"})
        assert len(mw._context_window) == 1
        assert "Action: edit" in mw._context_window[0]

    async def test_sliding_window_maxlen_5(self, middleware):
        mw, _, _ = middleware
        for i in range(7):
            mw.record_call(f"tool_{i}", {"arg": i})
        assert len(mw._context_window) == 5
        assert "tool_2" in mw._context_window[0]

    async def test_instinct_injection_when_matched(self, middleware):
        mw, store, manager = middleware
        await store.add_instinct_v2("test_project", r"edit", "Remember to test after edit", 0.9)
        await manager.load_project("test_project")
        reactions = mw.record_call("edit", {"file": "test.py"})
        assert reactions is not None
        assert "Remember to test after edit" in reactions[0]

    async def test_no_injection_when_no_match(self, middleware):
        mw, store, manager = middleware
        await store.add_instinct_v2("test_project", r"bug|error", "Fix the bug", 0.9)
        await manager.load_project("test_project")
        reactions = mw.record_call("edit", {"file": "test.py"})
        assert reactions is None

    async def test_inject_into_response(self, middleware):
        mw, _, _ = middleware
        result = {"status": "success", "formatted": "done"}
        modified = mw.inject_into_response(result, ["Check testing"])
        assert "MemoryMesh System Instincts" in modified["formatted"]

    async def test_get_tool_sequences(self, middleware):
        mw, _, _ = middleware
        mw.record_call("edit", {})
        mw.record_call("write", {})
        seqs = mw.get_tool_sequences()
        assert len(seqs) == 2

    async def test_reset_sequences(self, middleware):
        mw, _, _ = middleware
        mw.record_call("edit", {})
        mw.reset_sequences()
        assert len(mw.get_tool_sequences()) == 0
