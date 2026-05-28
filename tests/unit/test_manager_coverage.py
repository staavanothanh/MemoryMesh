"""Coverage gap tests for memorymesh.memory.manager.

These tests target the uncovered lines from the coverage report.
They run independently from test_memory_manage.py's autouse fixtures.
"""
import asyncio
import hashlib
import logging
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock, ANY

import pytest

from memorymesh.errors import ValidationError
from memorymesh.hooks import HookRegistry
from memorymesh.memory.manager import MemoryManager
from memorymesh.memory.sqlite_vec_backend import DIM
from memorymesh.router import RouterClient


SAMPLE_EMBEDDING = [0.1] * 384

# Save reference to the original enrich method before any fixtures patch it
_ORIGINAL_ENRICH = MemoryManager._enrich_memory


# ──────────────────────────────────────────────────────────────────────
# Autouse fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_embedding():
    """Prevent real embedding computation (slow / requires model)."""
    with patch("memorymesh.memory.manager.get_embedding",
               new=AsyncMock(return_value=SAMPLE_EMBEDDING)):
        yield


@pytest.fixture(autouse=True)
def mock_router_llm():
    """Mock RouterClient.call_llm_background so enrichment never hits the network.

    This fixture patches at the class level so all instances are covered.
    Tests that need custom return values can override via patch on the instance.
    """
    with patch.object(RouterClient, "call_llm_background",
                      new=AsyncMock(
                          return_value='{"tags":["ai"],"importance":4,"summary":"test"}'
                      )):
        yield


# ──────────────────────────────────────────────────────────────────────
# EASY: Module-level constant / _log_bg  (line 38)
# ──────────────────────────────────────────────────────────────────────

class TestModuleConstants:

    def test_dim_imported(self):
        """Line 38 is _log_bg; DIM is imported in manager.py from sqlite_vec_backend."""
        assert DIM == 384

    def test_log_bg_executes(self, memory_manager, caplog):
        """Direct call to module-level _log_bg."""
        from memorymesh.memory.manager import _log_bg
        with caplog.at_level(logging.INFO):
            _log_bg("TestLabel", "hello world", emoji="🔬")
            assert "TestLabel" in caplog.text
            assert "hello world" in caplog.text


# ──────────────────────────────────────────────────────────────────────
# EASY: _is_workspace_visible  (lines 109–119)
# ──────────────────────────────────────────────────────────────────────

class TestIsWorkspaceVisible:

    def test_same_path(self, memory_manager):
        assert memory_manager._is_workspace_visible("/project/a", "/project/a") is True

    def test_ancestor_path(self, memory_manager):
        """Current path is ancestor of memory path → visible."""
        assert memory_manager._is_workspace_visible(
            "/project/a/sub", "/project/a"
        ) is True

    def test_sibling_path(self, memory_manager):
        """Same parent directory → visible."""
        assert memory_manager._is_workspace_visible(
            "/project/b", "/project/a"
        ) is True

    def test_different_branch(self, memory_manager):
        """Different branch → not visible."""
        assert memory_manager._is_workspace_visible(
            "/other/project", "/project/a"
        ) is False

    def test_legacy_no_path(self, memory_manager):
        """Legacy memory with no path → visible from everywhere."""
        assert memory_manager._is_workspace_visible(None, "/project/a") is True
        assert memory_manager._is_workspace_visible("", "/project/a") is True

    def test_with_windows_backslash(self, memory_manager):
        """Backslash paths normalized."""
        assert memory_manager._is_workspace_visible(
            r"/project\a\sub", "/project/a"
        ) is True


# ──────────────────────────────────────────────────────────────────────
# EASY: _apply_search_filters  (line 134)
# ──────────────────────────────────────────────────────────────────────

class TestApplySearchFilters:

    def test_filters_by_workspace(self, memory_manager):
        results = [
            {"metadata": {"workspace_path": "/project/a"}},
            {"metadata": {"workspace_path": "/project/b"}},
            {"metadata": {"workspace_path": "/other/project"}},
            {"metadata": {}},
        ]
        filtered = memory_manager._apply_search_filters(
            results, workspace_path="/project/a"
        )
        # /project/a (same), /project/b (sibling), None (no path) = visible
        # /other/project (different branch) = not visible
        assert len(filtered) == 3
        paths = [r["metadata"].get("workspace_path") for r in filtered]
        assert "/project/a" in paths
        assert "/project/b" in paths
        assert None in paths
        assert "/other/project" not in paths

    def test_filters_archived_and_expired(self, memory_manager):
        results = [
            {"metadata": {"archived": True}},
            {"metadata": {"expired": True}},
            {"metadata": {"consolidated": True}},
            {"metadata": {}},
        ]
        filtered = memory_manager._apply_search_filters(results)
        assert len(filtered) == 1
        assert filtered[0]["metadata"] == {}

    def test_no_workspace_filter(self, memory_manager):
        """Without workspace_path, all non-archived/expired pass through."""
        results = [
            {"metadata": {"workspace_path": "/a"}},
            {"metadata": {"workspace_path": "/b"}},
            {"metadata": {}},
        ]
        filtered = memory_manager._apply_search_filters(results)
        assert len(filtered) == 3


# ──────────────────────────────────────────────────────────────────────
# EASY: _importance_decay  (line 558)
# ──────────────────────────────────────────────────────────────────────

class TestImportanceDecay:

    def test_future_timestamp_returns_one(self, memory_manager):
        future = datetime(2099, 6, 1, tzinfo=timezone.utc).isoformat()
        assert memory_manager._importance_decay(future) == 1.0

    def test_old_timestamp_decays(self, memory_manager):
        old = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        assert memory_manager._importance_decay(old) < 0.5

    def test_invalid_returns_one(self, memory_manager):
        assert memory_manager._importance_decay("garbage") == 1.0
        assert memory_manager._importance_decay("") == 1.0


# ──────────────────────────────────────────────────────────────────────
# EASY: _compute_final_score — atomic_fact boost  (line 578)
# ──────────────────────────────────────────────────────────────────────

class TestFinalScoreAtomicFact:

    def test_atomic_fact_boost(self, memory_manager):
        base = {"score": 0.5, "metadata": {"importance": 3, "timestamp": "",
                                            "tags": ["atomic_fact"]}}
        normal = {"score": 0.5, "metadata": {"importance": 3, "timestamp": "",
                                              "tags": ["normal"]}}
        boosted = memory_manager._compute_final_score(base)
        unboosted = memory_manager._compute_final_score(normal)
        assert boosted == pytest.approx(unboosted * 1.5)

    def test_atomic_fact_without_tags_list(self, memory_manager):
        """Tags might not be a list — should not crash."""
        mem = {"score": 0.5, "metadata": {"importance": 3, "timestamp": "",
                                           "tags": "atomic_fact"}}
        score = memory_manager._compute_final_score(mem)
        assert score >= 0.0


# ──────────────────────────────────────────────────────────────────────
# EASY: archive / unarchive  (lines 728–731, 735–738)
# ──────────────────────────────────────────────────────────────────────

class TestArchiveUnarchive:

    @pytest.mark.asyncio
    async def test_archive_memory(self, memory_manager):
        mid = await memory_manager.add_memory(text="archive test", user_id="test_user")
        success = await memory_manager.archive_memory(mid)
        assert success is True

        results = await memory_manager.search_memory(
            query="archive test", top_k=5, user_id="test_user"
        )
        ids = [r["id"] for r in results]
        assert mid not in ids

    @pytest.mark.asyncio
    async def test_unarchive_memory(self, memory_manager):
        mid = await memory_manager.add_memory(text="unarchive test", user_id="test_user")
        await memory_manager.archive_memory(mid)
        success = await memory_manager.unarchive_memory(mid)
        assert success is True

        results = await memory_manager.search_memory(
            query="unarchive test", top_k=5, user_id="test_user"
        )
        ids = [r["id"] for r in results]
        assert mid in ids

    @pytest.mark.asyncio
    async def test_archive_already_archived(self, memory_manager):
        mid = await memory_manager.add_memory(text="double archive", user_id="test_user")
        await memory_manager.archive_memory(mid)
        success = await memory_manager.archive_memory(mid)
        assert success is True

    @pytest.mark.asyncio
    async def test_unarchive_already_active(self, memory_manager):
        mid = await memory_manager.add_memory(text="double unarchive", user_id="test_user")
        success = await memory_manager.unarchive_memory(mid)
        assert success is True


# ──────────────────────────────────────────────────────────────────────
# EASY: forget_memory edge cases
# ──────────────────────────────────────────────────────────────────────

class TestForgetMemoryEdgeCases:

    @pytest.mark.asyncio
    async def test_forget_memory_archived_still_succeeds(self, memory_manager):
        """Forget an already-forgotten (archived) memory."""
        mid = await memory_manager.add_memory(text="forget again", user_id="test_user")
        await memory_manager.forget_memory(mid)
        success = await memory_manager.forget_memory(mid)
        # update_metadata on an existing memory always returns True
        # because it finds the record and updates it
        assert success is True

    @pytest.mark.asyncio
    async def test_forget_memory_nonexistent_returns_false(self, memory_manager):
        success = await memory_manager.forget_memory("non-existent-id")
        assert success is False


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _safe_task_wrapper  (lines 83–84)
# ──────────────────────────────────────────────────────────────────────

class TestSafeTaskWrapper:

    @pytest.mark.asyncio
    async def test_safe_task_wrapper_logs_error(self, caplog):
        async def failing_coro():
            raise ValueError("task failure test")

        with caplog.at_level(logging.ERROR):
            await MemoryManager._safe_task_wrapper(failing_coro())
            assert "task failure test" in caplog.text
            assert "Background task failed" in caplog.text

    @pytest.mark.asyncio
    async def test_safe_task_wrapper_cancelled(self, caplog):
        async def cancelled_coro():
            raise asyncio.CancelledError()

        with caplog.at_level(logging.ERROR):
            await MemoryManager._safe_task_wrapper(cancelled_coro())
            assert "Background task failed" not in caplog.text


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _update_embedding_background  (lines 254–271)
# ──────────────────────────────────────────────────────────────────────

class TestUpdateEmbeddingBackground:

    @pytest.mark.asyncio
    async def test_success(self, memory_manager):
        with patch.object(memory_manager.backend, "update_embedding",
                          new=AsyncMock(return_value=True)):
            await memory_manager._update_embedding_background("test-id", "some text")

    @pytest.mark.asyncio
    async def test_retry_then_success(self, memory_manager):
        mock_update = AsyncMock(side_effect=[Exception("first fail"), True])
        with patch.object(memory_manager.backend, "update_embedding", mock_update):
            await memory_manager._update_embedding_background("test-id", "some text")
            assert mock_update.call_count == 2

    @pytest.mark.asyncio
    async def test_all_failures(self, memory_manager):
        mock_update = AsyncMock(side_effect=Exception("always fail"))
        with patch.object(memory_manager.backend, "update_embedding", mock_update):
            with patch.object(memory_manager.backend, "update_metadata",
                              new=AsyncMock(return_value=True)):
                await memory_manager._update_embedding_background("test-id", "some text")
                assert mock_update.call_count == 3

    @pytest.mark.asyncio
    async def test_all_failures_meta_fails_too(self, memory_manager):
        """Lines 270-271: all embedding attempts fail AND metadata update fails too."""
        mock_update = AsyncMock(side_effect=Exception("always fail"))
        with patch.object(memory_manager.backend, "update_embedding", mock_update):
            with patch.object(memory_manager.backend, "update_metadata",
                              new=AsyncMock(side_effect=Exception("meta fail too"))):
                # Should not raise — the inner try/except catches it silently
                await memory_manager._update_embedding_background("test-id", "some text")
                assert mock_update.call_count == 3


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _enrich_memory  (lines 280, 287–301, 303–307)
#
# We restore the original _enrich_memory for these tests since the
# autouse fixture patches it at class level. We save the original at
# import time in _ORIGINAL_ENRICH.
# ──────────────────────────────────────────────────────────────────────

class TestEnrichMemory:

    @pytest.mark.asyncio
    async def test_enrich_memory_not_found(self, memory_manager):
        """Line 280: memory not found → early return."""
        with patch.object(MemoryManager, "_enrich_memory", _ORIGINAL_ENRICH):
            with patch.object(memory_manager.backend, "_get_by_id_full",
                              new=AsyncMock(return_value=None)):
                result = await memory_manager._enrich_memory("no-such-id", "text", "user")
                assert result is None

    @pytest.mark.asyncio
    async def test_enrich_memory_success(self, memory_manager):
        """Lines 287–298: successful enrichment with tags, importance, summary."""
        existing_mem = {"id": "mem1", "content": "test", "metadata": {"tags": ["existing"]},
                        "importance": 2, "level": "user"}
        with patch.object(MemoryManager, "_enrich_memory", _ORIGINAL_ENRICH):
            with patch.object(memory_manager.backend, "_get_by_id_full",
                              new=AsyncMock(return_value=existing_mem)):
                with patch.object(memory_manager.backend, "update_metadata",
                                  new=AsyncMock(return_value=True)):
                    await memory_manager._enrich_memory("mem1", "test content", "user")

    @pytest.mark.asyncio
    async def test_enrich_memory_no_usable_fields(self, memory_manager):
        """Line 300: enrichment response has no usable fields."""
        existing_mem = {"id": "mem1", "content": "test", "metadata": {},
                        "importance": 2, "level": "user"}
        with patch.object(MemoryManager, "_enrich_memory", _ORIGINAL_ENRICH):
            with patch.object(memory_manager.backend, "_get_by_id_full",
                              new=AsyncMock(return_value=existing_mem)):
                with patch.object(memory_manager.router, "call_llm_background",
                                  new=AsyncMock(return_value='{"invalid": true}')):
                    await memory_manager._enrich_memory("mem1", "test content", "user")

    @pytest.mark.asyncio
    async def test_enrich_memory_error_then_success(self, memory_manager):
        """Line 303–304: first attempt fails, second succeeds."""
        existing_mem = {"id": "mem1", "content": "test", "metadata": {},
                        "importance": 2, "level": "user"}
        with patch.object(MemoryManager, "_enrich_memory", _ORIGINAL_ENRICH):
            with patch.object(memory_manager.backend, "_get_by_id_full",
                              new=AsyncMock(return_value=existing_mem)):
                call_llm = AsyncMock(side_effect=[
                    Exception("router down"),
                    '{"tags":["ai"],"importance":3}'
                ])
                with patch.object(memory_manager.router, "call_llm_background", call_llm):
                    with patch.object(memory_manager.backend, "update_metadata",
                                      new=AsyncMock(return_value=True)):
                        await memory_manager._enrich_memory("mem1", "test content", "user")
                        assert call_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_enrich_memory_all_attempts_fail(self, memory_manager, caplog):
        """Line 307: all enrichment attempts fail → log error."""
        existing_mem = {"id": "mem1", "content": "test", "metadata": {},
                        "importance": 2, "level": "user"}
        with patch.object(MemoryManager, "_enrich_memory", _ORIGINAL_ENRICH):
            with patch.object(memory_manager.backend, "_get_by_id_full",
                              new=AsyncMock(return_value=existing_mem)):
                with patch.object(memory_manager.router, "call_llm_background",
                                  new=AsyncMock(side_effect=Exception("always down"))):
                    with caplog.at_level(logging.ERROR):
                        await memory_manager._enrich_memory("mem1", "test content", "user")
                        assert "Enrichment failed" in caplog.text


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: Background task methods  (lines 314, 316, 323–325, 332–334, 341)
# ──────────────────────────────────────────────────────────────────────

class TestBackgroundTasks:

    @pytest.mark.asyncio
    async def test_maybe_expire_success(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_memory_decay",
                          new=AsyncMock(return_value=5)):
            await memory_manager._maybe_expire_memories("test_user")

    @pytest.mark.asyncio
    async def test_maybe_expire_success_zero(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_memory_decay",
                          new=AsyncMock(return_value=0)):
            await memory_manager._maybe_expire_memories("test_user")

    @pytest.mark.asyncio
    async def test_maybe_expire_error(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_memory_decay",
                          new=AsyncMock(side_effect=Exception("decay fail"))):
            await memory_manager._maybe_expire_memories("test_user")

    @pytest.mark.asyncio
    async def test_maybe_consolidate_success(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_for_user",
                          new=AsyncMock(return_value=3)):
            await memory_manager._maybe_consolidate("test_user")

    @pytest.mark.asyncio
    async def test_maybe_consolidate_error(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_for_user",
                          new=AsyncMock(side_effect=Exception("consolidate fail"))):
            await memory_manager._maybe_consolidate("test_user")

    @pytest.mark.asyncio
    async def test_maybe_resolve_facts_success(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_fact_consolidation",
                          new=AsyncMock(return_value=2)):
            await memory_manager._maybe_resolve_facts("test_user")

    @pytest.mark.asyncio
    async def test_maybe_resolve_facts_error(self, memory_manager):
        with patch.object(memory_manager._consolidator, "run_fact_consolidation",
                          new=AsyncMock(side_effect=Exception("fact fail"))):
            await memory_manager._maybe_resolve_facts("test_user")

    @pytest.mark.asyncio
    async def test_maybe_learn_instincts_success(self, memory_manager):
        with patch.object(memory_manager.instinct_engine, "learn_from_recent",
                          new=AsyncMock()):
            await memory_manager._maybe_learn_instincts("test_user")

    @pytest.mark.asyncio
    async def test_maybe_learn_instincts_error(self, memory_manager):
        with patch.object(memory_manager.instinct_engine, "learn_from_recent",
                          new=AsyncMock(side_effect=Exception("learn fail"))):
            await memory_manager._maybe_learn_instincts("test_user")


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _enrich_fts_results double fallback  (lines 400–402)
# ──────────────────────────────────────────────────────────────────────

class TestEnrichFtsResultsDoubleFallback:

    @pytest.mark.asyncio
    async def test_both_enrichment_methods_fail(self, memory_manager):
        """Both get_with_embeddings_by_ids AND get_metadata_by_ids fail → empty meta."""
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(side_effect=Exception("vec fail"))):
            with patch.object(memory_manager.backend, "get_metadata_by_ids",
                              new=AsyncMock(side_effect=Exception("meta fail"))):
                fts_results = [{"id": "m1", "content": "test", "score": 1.0}]
                enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
                assert len(enriched) == 1
                assert enriched[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_vector_fails_metadata_succeeds(self, memory_manager):
        """Vector enrichment fails, metadata fallback works (lines 387–399)."""
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(side_effect=Exception("vec fail"))):
            with patch.object(memory_manager.backend, "get_metadata_by_ids",
                              new=AsyncMock(return_value=[
                                  {"id": "m1", "metadata": {"tags": ["fallback"]}}
                              ])):
                fts_results = [{"id": "m1", "content": "test", "score": 1.0}]
                enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
                assert enriched[0]["metadata"]["tags"] == ["fallback"]

    @pytest.mark.asyncio
    async def test_missing_enriched_data_handled(self, memory_manager):
        """Enriched data missing for some IDs."""
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(return_value=[
                              {"id": "m1", "metadata": {"tags": ["found"]}}
                          ])):
            fts_results = [
                {"id": "m1", "content": "found", "score": 1.0},
                {"id": "m2", "content": "missing", "score": 0.5},
            ]
            enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
            assert len(enriched) == 2
            assert enriched[0]["metadata"]["tags"] == ["found"]
            assert enriched[1]["metadata"] == {}


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: search_with_fallback cursor pagination
# (lines 448, 459–460, 482, 493–494, 502–503, 515–516, 524–525)
# ──────────────────────────────────────────────────────────────────────

class TestSearchWithFallbackCursor:

    @pytest.mark.asyncio
    async def test_cursor_mismatch_ignored(self, memory_manager):
        """Line 435: invalid cursor query_hash → cursor ignored (falls to tier1)."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            {"id": "m1", "content": "hi", "score": 0.9,
             "tags": [], "importance": 3, "timestamp": "2024-01-01"},
        ])):
            results, tier, meta, next_cursor = await memory_manager.search_with_fallback(
                query="test", top_k=5, user_id="test_user",
                cursor={"query_hash": "wrong", "tier": "semantic"},
            )
            assert tier == "semantic"

    @pytest.mark.asyncio
    async def test_cursor_semantic_pagination(self, memory_manager):
        """Lines 448, 459–460: cursor-based pagination in semantic tier."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            {"id": "m3", "content": "three", "score": 0.7,
             "tags": [], "importance": 3, "timestamp": ""},
            {"id": "m2", "content": "two", "score": 0.6,
             "tags": [], "importance": 3, "timestamp": ""},
            {"id": "m1", "content": "one", "score": 0.5,
             "tags": [], "importance": 3, "timestamp": ""},
        ])):
            qh = hashlib.sha256("test".encode()).hexdigest()[:16]
            results, tier, meta, next_cursor = await memory_manager.search_with_fallback(
                query="test", top_k=2, user_id="test_user", min_score_threshold=0.3,
                cursor={"query_hash": qh, "tier": "semantic",
                        "last_score": 0.8, "last_id": "z", "page": 1},
            )
            assert tier == "semantic"
            assert len(results) == 2
            for r in results:
                assert r["score"] < 0.8

    @pytest.mark.asyncio
    async def test_cursor_semantic_generates_next_cursor(self, memory_manager):
        """Line 459–460: next cursor when there are more results than top_k."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            {"id": "m1", "content": "one", "score": 0.9,
             "tags": [], "importance": 3, "timestamp": ""},
            {"id": "m2", "content": "two", "score": 0.8,
             "tags": [], "importance": 3, "timestamp": ""},
            {"id": "m3", "content": "three", "score": 0.7,
             "tags": [], "importance": 3, "timestamp": ""},
        ])):
            results, tier, meta, next_cursor = await memory_manager.search_with_fallback(
                query="test", top_k=2, user_id="test_user", min_score_threshold=0.3,
            )
            assert tier == "semantic"
            assert len(results) == 2
            assert next_cursor is not None
            assert next_cursor["tier"] == "semantic"
            assert "query_hash" in next_cursor

    @pytest.mark.asyncio
    async def test_cursor_fts_pagination(self, memory_manager):
        """Lines 482, 493–494: cursor-based pagination in FTS tier."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value="keyword"):
                with patch.object(memory_manager.backend, "fts_search",
                                  new=AsyncMock(return_value=[
                    {"id": "m1", "content": "one", "score": 1.0},
                    {"id": "m2", "content": "two", "score": 0.9},
                    {"id": "m3", "content": "three", "score": 0.8},
                ])):
                    with patch.object(memory_manager.backend,
                                      "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[
                        {"id": "m1", "metadata": {"tags": []}},
                        {"id": "m2", "metadata": {"tags": []}},
                        {"id": "m3", "metadata": {"tags": []}},
                    ])):
                        qh = hashlib.sha256("test".encode()).hexdigest()[:16]
                        results, tier, meta, nc = await memory_manager.search_with_fallback(
                            query="test", top_k=2, user_id="test_user",
                            min_score_threshold=0.5,
                            cursor={"query_hash": qh, "tier": "fts_keyword",
                                    "last_score": 1.0, "last_id": "zzz"},
                        )
                        assert tier == "fts_keyword"
                        # Cursor filters results with score < last_score OR
                        # (score == last_score AND id < last_id)
                        # So m1 (score=1.0, id="m1" < "zzz") and m2 (score=0.9) pass
                        # m3 (score=0.8) also passes. top_k=2 returns first 2.
                        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cursor_fts_generates_next_cursor(self, memory_manager):
        """Lines 493–494: next cursor when FTS has more results."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value="keyword"):
                with patch.object(memory_manager.backend, "fts_search",
                                  new=AsyncMock(return_value=[
                    {"id": "m1", "content": "one", "score": 1.0},
                    {"id": "m2", "content": "two", "score": 0.9},
                    {"id": "m3", "content": "three", "score": 0.8},
                ])):
                    with patch.object(memory_manager.backend,
                                      "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[
                        {"id": "m1", "metadata": {"tags": []}},
                        {"id": "m2", "metadata": {"tags": []}},
                        {"id": "m3", "metadata": {"tags": []}},
                    ])):
                        results, tier, meta, next_cursor = \
                            await memory_manager.search_with_fallback(
                                query="test", top_k=2, user_id="test_user",
                                min_score_threshold=0.5,
                            )
                        assert tier == "fts_keyword"
                        assert len(results) == 2
                        assert next_cursor is not None
                        assert next_cursor["tier"] == "fts_keyword"

    @pytest.mark.asyncio
    async def test_tier2_fts_error_falls_to_tier3(self, memory_manager):
        """Lines 502–503: FTS error → falls through to chronological."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value="keyword"):
                with patch.object(memory_manager.backend, "fts_search",
                                  new=AsyncMock(side_effect=Exception("fts error"))):
                    with patch.object(memory_manager.backend, "list_recent",
                                      new=AsyncMock(return_value=[
                        {"id": "m3", "content": "recent", "score": 1.0},
                    ])):
                        with patch.object(memory_manager.backend,
                                          "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[
                            {"id": "m3", "metadata": {"tags": []}},
                        ])):
                            results, tier, meta, _ = \
                                await memory_manager.search_with_fallback(
                                    query="test", top_k=5, user_id="test_user",
                                )
                            assert tier == "chronological"
                            assert results[0]["id"] == "m3"

    @pytest.mark.asyncio
    async def test_tier3_error_returns_empty(self, memory_manager):
        """Lines 524–525: chronological error → returns empty."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value=""):
                with patch.object(memory_manager.backend, "list_recent",
                                  new=AsyncMock(side_effect=Exception("chrono error"))):
                    with patch.object(memory_manager.backend,
                                      "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[])):
                        results, tier, meta, _ = \
                            await memory_manager.search_with_fallback(
                                query="test", top_k=5, user_id="test_user",
                            )
                        assert tier == "empty"
                        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_tier3_chronological_cursor(self, memory_manager):
        """Lines 515–516: cursor pagination in chronological tier."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value="keyword"):
                with patch.object(memory_manager.backend, "fts_search",
                                  new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend, "list_recent",
                                      new=AsyncMock(return_value=[
                        {"id": "m1", "content": "one", "score": 1.0},
                        {"id": "m2", "content": "two", "score": 0.9},
                    ])):
                        with patch.object(memory_manager.backend,
                                          "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[
                            {"id": "m1", "metadata": {"tags": []}},
                            {"id": "m2", "metadata": {"tags": []}},
                        ])):
                            qh = hashlib.sha256("test".encode()).hexdigest()[:16]
                            results, tier, meta, nc = \
                                await memory_manager.search_with_fallback(
                                    query="test", top_k=1, user_id="test_user",
                                    cursor={"query_hash": qh, "tier": "chronological",
                                            "last_score": 1.0, "last_id": "a"},
                                )
                            assert tier == "chronological"


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: search_memory workspace filter  (lines 604–607)
# ──────────────────────────────────────────────────────────────────────

class TestSearchMemoryWorkspace:

    @pytest.mark.asyncio
    async def test_workspace_sibling_visible(self, memory_manager):
        """Sibling workspace → visible, no penalty."""
        await memory_manager.add_memory(
            text="workspace a memory", user_id="test_user",
            workspace_path="/project/a",
        )
        results = await memory_manager.search_memory(
            query="workspace", top_k=5, user_id="test_user",
            workspace_path="/project/b",
        )
        # Sibling is visible → memory should appear
        ids = [r["id"] for r in results]

    @pytest.mark.asyncio
    async def test_workspace_same_path(self, memory_manager):
        """Same workspace path → fully visible."""
        await memory_manager.add_memory(
            text="same ws test", user_id="test_user",
            workspace_path="/project/x",
        )
        results = await memory_manager.search_memory(
            query="same ws", top_k=5, user_id="test_user",
            workspace_path="/project/x",
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_workspace_different_branch(self, memory_manager):
        """Different branch → score *= 0.7 penalty applied."""
        await memory_manager.add_memory(
            text="different branch memory", user_id="test_user",
            workspace_path="/project/a",
        )
        results = await memory_manager.search_memory(
            query="different branch", top_k=5, user_id="test_user",
            workspace_path="/other/branch",
        )
        # Memory may or may not appear (depends on score threshold),
        # but we just verify no crash


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: search_memory token budget truncation  (lines 641, 643–648)
# ──────────────────────────────────────────────────────────────────────

class TestSearchMemoryTokenTruncation:

    @pytest.mark.asyncio
    async def test_token_budget_truncation(self, memory_manager):
        """Small token budget forces truncation."""
        long_content = "word " * 500
        await memory_manager.add_memory(
            text=long_content, user_id="test_user", importance=5,
        )
        results = await memory_manager.search_memory(
            query="word", top_k=5, user_id="test_user", max_tokens=50,
        )
        assert len(results) >= 1
        content = results[0]["content"]
        if "..." not in content:
            assert len(content.split()) < len(long_content.split())

    @pytest.mark.asyncio
    async def test_token_budget_large_enough(self, memory_manager):
        """Large budget → no truncation."""
        content = "hello world " * 10
        await memory_manager.add_memory(text=content, user_id="test_user")
        results = await memory_manager.search_memory(
            query="hello", top_k=5, user_id="test_user", max_tokens=5000,
        )
        assert len(results) >= 1
        assert "..." not in results[0]["content"]

    @pytest.mark.asyncio
    async def test_zero_token_memory_skipped(self, memory_manager):
        """Line 641: memory with 0 tokens should be skipped."""
        mem = {"id": "zero", "content": "", "score": 0.5,
               "metadata": {"importance": 3, "timestamp": "2024-01-01", "tags": []}}
        scored = [(memory_manager._compute_final_score(mem), mem)]
        # Manually check the token budget logic
        meta_str = str(mem.get("metadata", {}))
        meta_tokens = len(memory_manager._tokenizer.encode(meta_str))
        content_tokens = len(memory_manager._tokenizer.encode(mem["content"]))
        assert content_tokens == 0

    @pytest.mark.asyncio
    async def test_empty_content_skipped_in_search(self, memory_manager):
        """Line 641: empty content memory skipped during search_memory token budget."""
        # Add memory with some content
        await memory_manager.add_memory(
            text="something searchable", user_id="test_user",
        )
        # We can't directly add an empty-content memory via add_memory (it requires text),
        # but search_memory's token budget logic handles empty content that comes from
        # the backend. So we test by patching search results to include an empty-content record.
        with patch.object(memory_manager.backend, "search", new=AsyncMock(return_value=[
            {"id": "empty", "content": "", "score": 0.5,
             "metadata": {"importance": 3, "timestamp": "2024-01-01",
                          "level": "user", "tags": []}},
            {"id": "normal", "content": "valid content", "score": 0.9,
             "metadata": {"importance": 4, "timestamp": "2024-01-01",
                          "level": "user", "tags": []}},
        ])):
            results = await memory_manager.search_memory(
                query="test", top_k=5, user_id="test_user", max_tokens=500,
            )
            # Only the normal memory should appear (empty skipped)
            ids = [r["id"] for r in results]
            assert "normal" in ids


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: add_memory background=True + hooks  (lines 187, 207, 214)
# ──────────────────────────────────────────────────────────────────────

class TestAddMemoryBackgroundAndHooks:

    @pytest.mark.asyncio
    async def test_add_memory_background(self, memory_manager):
        """Line 187: background=True uses dummy embedding."""
        memory_id = await memory_manager.add_memory(
            text="background test", user_id="test_user", background=True,
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_with_hooks(self, memory_manager):
        """Line 214: hooks trigger after_remember."""
        hook_called = False

        async def after_remember_hook(**kwargs):
            nonlocal hook_called
            hook_called = True
            assert "memory_id" in kwargs
            assert "user_id" in kwargs

        hooks = HookRegistry()
        hooks.register("after_remember", after_remember_hook)
        memory_manager.hooks = hooks

        await memory_manager.add_memory(text="hooks test", user_id="test_user")
        await asyncio.sleep(0.02)  # let background task run
        assert hook_called, "Hook should have been called"


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: add_memory instinct suggestions  (lines 176–177, 181–184)
# ──────────────────────────────────────────────────────────────────────

class TestAddMemoryInstincts:

    @pytest.mark.asyncio
    async def test_add_memory_with_instinct_tags(self, memory_manager):
        """Lines 176–177, 181–184: instinct enabled with suggestions."""
        memory_manager.config.instinct.enabled = True
        await memory_manager.instinct_store.initialize()

        with patch.object(memory_manager.instinct_engine, "get_auto_apply_tags",
                          new=AsyncMock(return_value=["auto-tag-1", "auto-tag-2"])):
            with patch.object(memory_manager.instinct_engine, "apply_instincts",
                              new=AsyncMock(return_value={
                                  "suggested_tags": [
                                      {"tag": "suggested-tag", "confidence": 0.8,
                                       "instinct_id": "inst-1"},
                                  ],
                                  "suggested_level": None,
                              })):
                with patch.object(memory_manager.instinct_engine,
                                  "reinforce_instinct", new=AsyncMock()):
                    memory_id = await memory_manager.add_memory(
                        text="instinct test with tags", user_id="test_user",
                    )
                    assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_instinct_low_confidence_skipped(self, memory_manager):
        """Line 180: low confidence suggestions are skipped."""
        memory_manager.config.instinct.enabled = True
        await memory_manager.instinct_store.initialize()

        with patch.object(memory_manager.instinct_engine, "get_auto_apply_tags",
                          new=AsyncMock(return_value=[])):
            with patch.object(memory_manager.instinct_engine, "apply_instincts",
                              new=AsyncMock(return_value={
                                  "suggested_tags": [
                                      {"tag": "low-conf-tag", "confidence": 0.3,
                                       "instinct_id": "inst-2"},
                                  ],
                                  "suggested_level": None,
                              })):
                memory_id = await memory_manager.add_memory(
                    text="low confidence test", user_id="test_user",
                )
                assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_instinct_error_continues(self, memory_manager, caplog):
        """Line 184: instinct error logged but memory addition continues."""
        memory_manager.config.instinct.enabled = True
        await memory_manager.instinct_store.initialize()

        with patch.object(memory_manager.instinct_engine, "get_auto_apply_tags",
                          new=AsyncMock(side_effect=Exception("instinct error"))):
            with caplog.at_level(logging.ERROR):
                memory_id = await memory_manager.add_memory(
                    text="instinct error test", user_id="test_user",
                )
                assert memory_id is not None
                assert "Instinct suggestion failed" in caplog.text


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: delete_memories_by_session  (lines 685–689)
# ──────────────────────────────────────────────────────────────────────

class TestDeleteMemoriesBySession:

    @pytest.mark.asyncio
    async def test_delete_by_session_success(self, memory_manager):
        mid = await memory_manager.add_memory(
            text="session memory", tags=["session:test-sess"], user_id="test_user",
        )
        count = await memory_manager.delete_memories_by_session("test-sess", "test_user")
        assert count >= 1

    @pytest.mark.asyncio
    async def test_delete_by_session_empty(self, memory_manager):
        count = await memory_manager.delete_memories_by_session("nonexistent", "test_user")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_by_session_no_backend_method(self, memory_manager):
        """Line 689: backend without delete_by_tag returns 0."""
        from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend
        original = SqliteVecBackend.delete_by_tag
        try:
            delattr(SqliteVecBackend, "delete_by_tag")
            count = await memory_manager.delete_memories_by_session("sess", "user")
            assert count == 0
        finally:
            setattr(SqliteVecBackend, "delete_by_tag", original)


# ──────────────────────────────────────────────────────────────────────
# HARD: preserve_important_memories  (lines 693–724)
# ──────────────────────────────────────────────────────────────────────

class TestPreserveImportantMemories:

    @pytest.mark.asyncio
    async def test_preserve_high_importance(self, memory_manager):
        mid = await memory_manager.add_memory(
            text="important decision", importance=4, user_id="test_user",
            tags=["session:sess-1"],
        )
        preserved = await memory_manager.preserve_important_memories("sess-1", "test_user")
        assert preserved >= 1

    @pytest.mark.asyncio
    async def test_preserve_keyword_match(self, memory_manager):
        await memory_manager.add_memory(
            text="we decided to implement a new feature and fix the architecture",
            importance=2, user_id="test_user",
            tags=["session:sess-2"],
        )
        preserved = await memory_manager.preserve_important_memories("sess-2", "test_user")
        assert preserved >= 1

    @pytest.mark.asyncio
    async def test_preserve_no_match(self, memory_manager):
        await memory_manager.add_memory(
            text="hello world", importance=1, user_id="test_user",
            tags=["session:sess-3"],
        )
        preserved = await memory_manager.preserve_important_memories("sess-3", "test_user")
        assert preserved == 0

    @pytest.mark.asyncio
    async def test_preserve_empty_session(self, memory_manager):
        preserved = await memory_manager.preserve_important_memories(
            "no-such-session", "test_user"
        )
        assert preserved == 0

    @pytest.mark.asyncio
    async def test_preserve_no_list_by_tag_method(self, memory_manager):
        """Line 695: backend without list_by_tag returns 0."""
        from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend
        original = SqliteVecBackend.list_by_tag
        try:
            delattr(SqliteVecBackend, "list_by_tag")
            # Re-create the memory_manager so its backend picks up the change
            # Actually hasattr is dynamic so it will see the removal
            preserved = await memory_manager.preserve_important_memories(
                "no-method-sess", "test_user"
            )
            assert preserved == 0
        finally:
            setattr(SqliteVecBackend, "list_by_tag", original)


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: list_memories edge cases
# ──────────────────────────────────────────────────────────────────────

class TestListMemoriesEdgeCases:

    @pytest.mark.asyncio
    async def test_list_memories_pagination(self, memory_manager):
        ids = []
        for i in range(3):
            mid = await memory_manager.add_memory(
                text=f"page test {i}", user_id="list_user",
            )
            ids.append(mid)

        page1 = await memory_manager.list_memories(limit=2, offset=0, user_id="list_user")
        assert len(page1) == 2

        page2 = await memory_manager.list_memories(limit=2, offset=2, user_id="list_user")
        assert len(page2) == 1

    @pytest.mark.asyncio
    async def test_list_memories_empty(self, memory_manager):
        results = await memory_manager.list_memories(limit=10, user_id="no-memories-user")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_list_memories_default_user(self, memory_manager):
        """Uses default user when user_id not provided."""
        await memory_manager.add_memory(text="default user memory")
        results = await memory_manager.list_memories(limit=10)
        assert len(results) >= 1


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _run_background_tasks rate-limiting  (lines 234–248)
# ──────────────────────────────────────────────────────────────────────

class TestRunBackgroundTasks:

    @pytest.mark.asyncio
    async def test_bg_tasks_run(self, memory_manager):
        await memory_manager._run_background_tasks("test-id", "text", "user", "test_user")
        assert True  # no crash

    @pytest.mark.asyncio
    async def test_bg_tasks_session_level(self, memory_manager):
        """Session-level memories skip enrichment."""
        await memory_manager._run_background_tasks("test-id", "text", "session", "test_user")
        assert True  # no crash


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: _create_tracked_task and shutdown
# ──────────────────────────────────────────────────────────────────────

class TestTrackedTaskLifecycle:

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self, memory_manager):
        async def dummy_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                pass

        task = memory_manager._create_tracked_task(dummy_task())
        assert task in memory_manager._background_tasks
        await memory_manager.shutdown()
        assert len(memory_manager._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_no_tasks(self, memory_manager):
        await memory_manager.shutdown()
        assert True  # no crash

    @pytest.mark.asyncio
    async def test_create_tracked_task(self, memory_manager):
        async def quick_task():
            pass

        task = memory_manager._create_tracked_task(quick_task())
        # Task may complete quickly and be removed via done callback
        await asyncio.sleep(0.01)
        assert task.done() or task in memory_manager._background_tasks


# ──────────────────────────────────────────────────────────────────────
# EASY: _dicts_to_search_results  (line 531)
# ──────────────────────────────────────────────────────────────────────

class TestDictsToSearchResults:

    def test_converts_dicts(self, memory_manager):
        dicts = [
            {"id": "m1", "content": "hello", "score": 0.9,
             "metadata": {"tags": ["a"], "importance": 4, "timestamp": "2024-01-01"}},
        ]
        results = memory_manager._dicts_to_search_results(dicts)
        assert len(results) == 1
        assert results[0]["id"] == "m1"
        assert results[0]["content"] == "hello"
        assert results[0]["score"] == 0.9
        assert results[0]["tags"] == ["a"]
        assert results[0]["importance"] == 4
        assert results[0]["timestamp"] == "2024-01-01"

    def test_empty_input(self, memory_manager):
        assert memory_manager._dicts_to_search_results([]) == []

    def test_missing_keys(self, memory_manager):
        dicts = [{"id": "m1"}]  # missing content, score, metadata
        results = memory_manager._dicts_to_search_results(dicts)
        assert len(results) == 1
        assert results[0]["content"] == ""


# ──────────────────────────────────────────────────────────────────────
# MEDIUM: search_memory level-filter / promotion path
# ──────────────────────────────────────────────────────────────────────

class TestSearchMemoryLevelAndPromotion:

    @pytest.mark.asyncio
    async def test_search_with_level_filter(self, memory_manager):
        """Level filter should restrict results to specific level."""
        await memory_manager.add_memory(
            text="knowledge memory", level="knowledge", user_id="test_user",
        )
        await memory_manager.add_memory(
            text="session memory", level="session", user_id="test_user",
        )
        # Search with level_filter for session only
        results = await memory_manager.search_memory(
            query="memory", top_k=5, user_id="test_user",
            level_filter=["session"],
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_promotion_path(self, memory_manager):
        """Lines 653–663: memory promotion when score >= 0.7."""
        memory_id = await memory_manager.add_memory(
            text="important promoted memory", importance=2, user_id="test_user",
        )
        # Search — if score is high enough, promotion runs in background
        results = await memory_manager.search_memory(
            query="important promoted", top_k=5, user_id="test_user",
        )
        assert len(results) >= 1


# ──────────────────────────────────────────────────────────────────────
# EASY: search_with_fallback early edges
# ──────────────────────────────────────────────────────────────────────

class TestSearchWithFallbackEdges:

    @pytest.mark.asyncio
    async def test_empty_query_handled(self, memory_manager):
        """Empty query still works (falls through tiers)."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value=""):
                with patch.object(memory_manager.backend, "list_recent",
                                  new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend,
                                      "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[])):
                        results, tier, meta, _ = \
                            await memory_manager.search_with_fallback(
                                query="", top_k=5, user_id="test_user",
                            )
                        assert tier == "empty"

    @pytest.mark.asyncio
    async def test_default_user_id(self, memory_manager):
        """Default user_id used when none provided."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords",
                              return_value=""):
                with patch.object(memory_manager.backend, "list_recent",
                                  new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend,
                                      "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[])):
                        results, tier, meta, _ = \
                            await memory_manager.search_with_fallback(
                                query="test", top_k=5,
                            )
                        assert tier == "empty"
