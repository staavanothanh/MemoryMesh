import pytest
import pytest_asyncio
import aiosqlite
import re

from memorymesh.memory.instinct_store import InstinctStore
from memorymesh.memory.instinct_manager import InstinctManager, CompiledInstinct, background_learning_daemon


@pytest_asyncio.fixture
async def store():
    s = InstinctStore(":memory:")
    s._db = await aiosqlite.connect(":memory:")
    s._db.row_factory = aiosqlite.Row
    await s._db.execute("PRAGMA journal_mode=WAL")
    await s._db.execute("""
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
    await s._db.execute("""
        CREATE INDEX IF NOT EXISTS idx_instincts_v2_project
        ON instincts_v2(project_id, confidence_score DESC)
    """)
    await s._db.commit()
    yield s
    await s._db.close()


@pytest_asyncio.fixture
async def manager(store):
    m = InstinctManager(store)
    yield m


class TestInstinctManager:
    async def test_load_all_empty(self, manager):
        await manager.load_all()
        assert len(manager._cache) == 0

    async def test_add_and_load(self, manager, store):
        await store.add_instinct_v2("proj1", "test", "found it", 0.9)
        await manager.load_all()
        assert "proj1" in manager._cache
        assert len(manager._cache["proj1"]) == 1
        assert isinstance(manager._cache["proj1"][0].regex, re.Pattern)

    async def test_evaluate_match(self, manager, store):
        await store.add_instinct_v2("proj1", r"bug|error|fail", "Check error handling", 0.8)
        await manager.load_all()
        reactions = manager.evaluate("proj1", "found a bug in the code")
        assert len(reactions) == 1
        assert "Check error handling" in reactions[0]

    async def test_evaluate_no_match(self, manager, store):
        await store.add_instinct_v2("proj1", r"bug|error|fail", "Check error handling", 0.8)
        await manager.load_all()
        reactions = manager.evaluate("proj1", "everything is fine")
        assert len(reactions) == 0

    async def test_evaluate_wrong_project(self, manager, store):
        await store.add_instinct_v2("proj1", r"test", "reaction", 0.8)
        await manager.load_all()
        reactions = manager.evaluate("proj2", "test output")
        assert len(reactions) == 0

    async def test_hot_reload_project(self, manager, store):
        await store.add_instinct_v2("proj1", r"first", "first reaction", 0.7)
        await manager.load_all()
        assert len(manager._cache["proj1"]) == 1
        await store.add_instinct_v2("proj1", r"second", "second reaction", 0.9)
        await manager.load_project("proj1")
        assert len(manager._cache["proj1"]) == 2
        assert manager._cache["proj1"][0].confidence == 0.9  # sorted desc

    async def test_bad_regex_skipped(self, manager, store):
        await store.add_instinct_v2("proj1", r"[invalid", "bad regex", 0.5)
        await manager.load_all()
        assert "proj1" not in manager._cache or len(manager._cache["proj1"]) == 0


class TestExtractNgrams:
    def test_empty_sequences(self):
        result = InstinctManager.extract_ngrams([], threshold=2)
        assert result == []

    def test_short_sequences(self):
        result = InstinctManager.extract_ngrams(["edit write test"], threshold=1)
        assert len(result) > 0
        assert "edit" in result[0]["trigger_regex"]

    def test_threshold_filter(self):
        seqs = ["edit write test"] * 5
        result = InstinctManager.extract_ngrams(seqs, threshold=3)
        assert len(result) > 0
        for r in result:
            assert r["confidence_score"] > 0.2

    def test_confidence_never_exceeds_09(self):
        seqs = ["edit write"] * 100
        result = InstinctManager.extract_ngrams(seqs, threshold=1)
        for r in result:
            assert r["confidence_score"] <= 0.9
