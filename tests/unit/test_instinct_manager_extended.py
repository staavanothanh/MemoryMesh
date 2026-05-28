"""Extended tests for instinct_manager.py — is_safe_regex, dedup, clear, background daemon."""

import pytest
import pytest_asyncio
import aiosqlite
from unittest.mock import AsyncMock, patch

from memorymesh.memory.instinct_store import InstinctStore
from memorymesh.memory.instinct_manager import InstinctManager, background_learning_daemon


@pytest_asyncio.fixture
async def store():
    s = InstinctStore(":memory:")
    s._db = await aiosqlite.connect(":memory:")
    s._db.row_factory = aiosqlite.Row
    await s._db.executescript("""
        CREATE TABLE IF NOT EXISTS instincts_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            trigger_regex TEXT NOT NULL,
            reaction TEXT NOT NULL,
            confidence_score REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_instincts_v2_project ON instincts_v2(project_id, confidence_score DESC);
    """)
    await s._db.commit()
    yield s
    await s._db.close()


@pytest_asyncio.fixture
async def manager(store):
    m = InstinctManager(store)
    yield m


class TestIsSafeRegex:
    def test_too_long(self):
        assert not InstinctManager.is_safe_regex("a" * 201, max_length=200)

    def test_catastrophic_pattern(self):
        assert not InstinctManager.is_safe_regex(r"(.+)+")
        assert not InstinctManager.is_safe_regex(r"(?:[a-z])+")

    def test_safe_pattern(self):
        assert InstinctManager.is_safe_regex(r"\d+")
        assert InstinctManager.is_safe_regex(r"bug|error|fail")
        assert InstinctManager.is_safe_regex(r"hello world")


class TestDedupInstincts:
    def test_empty_list(self):
        result = InstinctManager._dedup_instincts([], 0.95)
        assert result == []

    def test_dedup_keeps_highest_confidence(self):
        records = [
            {"trigger_regex": "test", "reaction": "low", "confidence_score": 0.5},
            {"trigger_regex": "test", "reaction": "high", "confidence_score": 0.9},
        ]
        result = InstinctManager._dedup_instincts(records)
        assert len(result) == 1
        assert result[0]["reaction"] == "high"

    def test_different_patterns_kept_separate(self):
        records = [
            {"trigger_regex": "pattern_a", "reaction": "r1", "confidence_score": 0.5},
            {"trigger_regex": "pattern_b", "reaction": "r2", "confidence_score": 0.7},
        ]
        result = InstinctManager._dedup_instincts(records)
        assert len(result) == 2


class TestClear:
    def test_empty_cache(self, manager):
        manager.clear()
        assert len(manager._cache) == 0

    async def test_clears_cached_data(self, manager, store):
        await store.add_instinct_v2("proj1", r"test", "reaction", 0.8)
        await manager.load_all()
        assert "proj1" in manager._cache
        manager.clear()
        assert len(manager._cache) == 0


class TestLoadAll:
    @pytest.mark.asyncio
    async def test_skips_unsafe_regex(self, manager, store):
        """load_all should skip instincts with unsafe regex patterns."""
        await store.add_instinct_v2("proj1", r"safe", "good", 0.8)
        await store.add_instinct_v2("proj1", r"(.+)+", "bad", 0.9)
        await manager.load_all()
        assert "proj1" in manager._cache
        assert len(manager._cache["proj1"]) == 1
        assert manager._cache["proj1"][0].reaction == "good"

    @pytest.mark.asyncio
    async def test_skips_invalid_regex(self, manager, store):
        """load_all should skip instincts with regex that raise re.error."""
        await store.add_instinct_v2("proj1", r"[valid", "should skip", 0.9)
        await manager.load_all()
        if "proj1" in manager._cache:
            assert len(manager._cache["proj1"]) == 0


class TestEvaluate:
    def test_no_project_in_cache(self, manager):
        reactions = manager.evaluate("nonexistent", "test text")
        assert reactions == []

    def test_below_confidence_floor_skipped(self, manager):
        manager._cache["proj1"] = [
            type("Inst", (), {"confidence": 0.1, "regex": type("R", (), {"search": lambda s, t: None})(), "reaction": "low"})()
        ]
        reactions = manager.evaluate("proj1", "test")
        assert reactions == []


class TestBackgroundLearningDaemon:
    @pytest.mark.asyncio
    async def test_happy_path_adds_instincts(self, manager, store):
        sequences = ["edit write test"] * 3
        await background_learning_daemon(manager, store, sequences, "test_project")
        instincts = await store.get_instincts_v2("test_project")
        assert len(instincts) >= 1

    @pytest.mark.asyncio
    async def test_unsafe_regex_skipped(self, manager, store):
        """Background learning should skip unsafe regex patterns."""
        with patch.object(InstinctManager, "extract_ngrams", return_value=[
            {"trigger_regex": "(.+)+", "reaction": "bad", "confidence_score": 0.5},
        ]):
            await background_learning_daemon(manager, store, [""] * 3, "test_project")
            instincts = await store.get_instincts_v2("test_project")
            assert len(instincts) == 0

    @pytest.mark.asyncio
    async def test_store_error_handled(self, manager, store):
        """When store.add_instinct_v2 raises, it should be caught gracefully."""
        store.add_instinct_v2 = AsyncMock(side_effect=Exception("DB error"))
        with patch.object(InstinctManager, "extract_ngrams", return_value=[
            {"trigger_regex": r"\d+", "reaction": "test", "confidence_score": 0.5},
        ]):
            await background_learning_daemon(manager, store, [""] * 3, "test_project")
            # Should not raise

    @pytest.mark.asyncio
    async def test_extract_ngrams_returns_none_skips(self, manager, store):
        """When extract_ngrams returns empty list, nothing should be added."""
        with patch.object(InstinctManager, "extract_ngrams", return_value=[]):
            await background_learning_daemon(manager, store, [], "test_project")
            instincts = await store.get_instincts_v2("test_project")
            assert len(instincts) == 0
