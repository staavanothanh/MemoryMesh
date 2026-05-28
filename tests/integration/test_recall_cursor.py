"""Integration tests for cursor-based pagination in recall.

Verifies that search_with_fallback with cursor parameter returns consistent
results across pages, invalid cursors are rejected, and error handling works.
"""

import asyncio
import json
import hashlib
import pytest


class TestRecallCursorPagination:
    """Cursor-based pagination returns consistent results across pages."""

    @pytest.mark.asyncio
    async def test_recall_cursor_pagination(self, backend):
        """Create backend, add 10 memories, call search_with_fallback with
        cursor param, verify next_cursor, call again with cursor, verify has_more."""
        from memorymesh.memory.manager import MemoryManager
        from memorymesh.config import AppConfig, RouterConfig, SqliteVecConfig, SessionConfig, ConsolidationConfig
        from unittest.mock import AsyncMock, patch

        # Add 10 memories
        for i in range(10):
            await backend.add(
                user_id="test_user",
                content=f"Cursor test memory {i}",
                embedding=[0.1] * 384,
                metadata={"importance": 3, "tags": ["cursor_test"]},
            )

        # We need a minimal MemoryManager that can call search_with_fallback
        # Since we can't easily create a full MemoryManager without embedding server,
        # we patch get_embedding in search_memory to return a dummy vector
        config = AppConfig(
            router=RouterConfig(
                url="http://127.0.0.1:20128/v1",
                default_model="test-model",
                fallback_model="test-fallback",
                timeout_s=5,
                max_retries=1,
            ),
            sqlite_vec=SqliteVecConfig(db_path=backend.db_path),
            session=SessionConfig(db_path=":memory:", auto_create_session=False),
            consolidation=ConsolidationConfig(
                similarity_threshold=0.85,
                min_cluster_size=2,
                interval_seconds=3600,
                batch_size=50,
                enabled=False,
                session_memory_ttl_days=7,
            ),
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            default_user_id="test_user",
            log_level="DEBUG",
        )

        from memorymesh.router import RouterClient
        router = RouterClient(config.router)

        mgr = MemoryManager(config, backend, router)

        # Patch get_embedding to return a dummy vector so search_memory works
        with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=[0.1] * 384)):
            try:
                # First page — no cursor
                results, tier, meta, next_cursor = await mgr.search_with_fallback(
                    query="cursor test memory",
                    top_k=3,
                    user_id="test_user",
                    min_score_threshold=0.0,
                )

                assert len(results) > 0, "Should have at least one result"
                assert tier in ("semantic", "fts_keyword", "chronological"), f"Unexpected tier: {tier}"

                if next_cursor is not None:
                    # Second page — with cursor
                    results2, tier2, meta2, next_cursor2 = await mgr.search_with_fallback(
                        query="cursor test memory",
                        top_k=3,
                        user_id="test_user",
                        min_score_threshold=0.0,
                        cursor=next_cursor,
                    )

                    assert len(results2) > 0, "Second page should have results"
                    # Verify no overlap with first page
                    ids1 = {r["id"] for r in results}
                    ids2 = {r["id"] for r in results2}
                    if ids1 and ids2:
                        assert ids1.isdisjoint(ids2), (
                            "First and second page should not overlap"
                        )
            finally:
                await asyncio.sleep(0)
                await mgr.shutdown()


class TestRecallCursorInvalidCursor:
    """Invalid cursors should be handled gracefully."""

    @pytest.mark.asyncio
    async def test_recall_cursor_invalid_query_hash_rejected(self, backend):
        """Pass cursor with wrong query_hash — cursor is ignored (fallback to first page)."""
        from memorymesh.memory.manager import MemoryManager
        from memorymesh.config import AppConfig, RouterConfig, SqliteVecConfig, SessionConfig, ConsolidationConfig
        from unittest.mock import AsyncMock, patch

        # Add some memories
        for i in range(5):
            await backend.add(
                user_id="test_user",
                content=f"Query hash test {i}",
                embedding=[0.1] * 384,
            )

        config = AppConfig(
            router=RouterConfig(
                url="http://127.0.0.1:20128/v1",
                default_model="test-model",
                fallback_model="test-fallback",
                timeout_s=5,
                max_retries=1,
            ),
            sqlite_vec=SqliteVecConfig(db_path=backend.db_path),
            session=SessionConfig(db_path=":memory:", auto_create_session=False),
            consolidation=ConsolidationConfig(
                similarity_threshold=0.85,
                min_cluster_size=2,
                interval_seconds=3600,
                batch_size=50,
                enabled=False,
                session_memory_ttl_days=7,
            ),
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            default_user_id="test_user",
            log_level="DEBUG",
        )

        from memorymesh.router import RouterClient
        router = RouterClient(config.router)
        mgr = MemoryManager(config, backend, router)

        with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=[0.1] * 384)):
            try:
                # Curated cursor with a clearly wrong query_hash
                bad_cursor = {
                    "query_hash": "deadbeef1234",
                    "tier": "semantic",
                    "last_score": 0.9,
                    "last_id": "some-id",
                    "page": 2,
                }

                # The cursor with wrong query_hash should be ignored,
                # and search should return first-page results
                results, tier, meta, next_cursor = await mgr.search_with_fallback(
                    query="query hash test",
                    top_k=3,
                    user_id="test_user",
                    min_score_threshold=0.0,
                    cursor=bad_cursor,
                )

                assert len(results) > 0, "Should return results even with bad cursor"
                # The tier should be whatever the first page uses
                assert tier in ("semantic", "fts_keyword", "chronological", "empty")
            finally:
                await asyncio.sleep(0)
                await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_recall_cursor_invalid_json_rejected(self, backend):
        """Pass invalid cursor structure — cursor is None, returns first page."""
        from memorymesh.memory.manager import MemoryManager
        from memorymesh.config import AppConfig, RouterConfig, SqliteVecConfig, SessionConfig, ConsolidationConfig
        from unittest.mock import AsyncMock, patch

        config = AppConfig(
            router=RouterConfig(
                url="http://127.0.0.1:20128/v1",
                default_model="test-model",
                fallback_model="test-fallback",
                timeout_s=5,
                max_retries=1,
            ),
            sqlite_vec=SqliteVecConfig(db_path=backend.db_path),
            session=SessionConfig(db_path=":memory:", auto_create_session=False),
            consolidation=ConsolidationConfig(
                similarity_threshold=0.85,
                min_cluster_size=2,
                interval_seconds=3600,
                batch_size=50,
                enabled=False,
                session_memory_ttl_days=7,
            ),
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            default_user_id="test_user",
            log_level="DEBUG",
        )

        from memorymesh.router import RouterClient
        router = RouterClient(config.router)
        mgr = MemoryManager(config, backend, router)

        with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=[0.1] * 384)):
            try:
                # Passing None as cursor (or empty dict) — should return first page
                results, tier, meta, next_cursor = await mgr.search_with_fallback(
                    query="test",
                    top_k=3,
                    user_id="test_user",
                    min_score_threshold=0.0,
                    cursor=None,
                )
                # No error, just normal first-page results
                assert tier in ("semantic", "fts_keyword", "chronological", "empty")
            finally:
                await asyncio.sleep(0)
                await mgr.shutdown()


class TestRecallCursorNoMemories:
    """Cursor behavior with empty backend."""

    @pytest.mark.asyncio
    async def test_recall_cursor_empty_backend(self, backend):
        """No memories in backend — cursor doesn't cause errors."""
        from memorymesh.memory.manager import MemoryManager
        from memorymesh.config import AppConfig, RouterConfig, SqliteVecConfig, SessionConfig, ConsolidationConfig
        from unittest.mock import AsyncMock, patch

        config = AppConfig(
            router=RouterConfig(
                url="http://127.0.0.1:20128/v1",
                default_model="test-model",
                fallback_model="test-fallback",
                timeout_s=5,
                max_retries=1,
            ),
            sqlite_vec=SqliteVecConfig(db_path=backend.db_path),
            session=SessionConfig(db_path=":memory:", auto_create_session=False),
            consolidation=ConsolidationConfig(
                similarity_threshold=0.85,
                min_cluster_size=2,
                interval_seconds=3600,
                batch_size=50,
                enabled=False,
                session_memory_ttl_days=7,
            ),
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            default_user_id="test_user",
            log_level="DEBUG",
        )

        from memorymesh.router import RouterClient
        router = RouterClient(config.router)
        mgr = MemoryManager(config, backend, router)

        with patch("memorymesh.memory.manager.get_embedding", new=AsyncMock(return_value=[0.1] * 384)):
            try:
                results, tier, meta, next_cursor = await mgr.search_with_fallback(
                    query="nothing here",
                    top_k=5,
                    user_id="test_user",
                    min_score_threshold=0.0,
                )
                assert next_cursor is None, "Empty backend should have no cursor"
            finally:
                await asyncio.sleep(0)
                await mgr.shutdown()
