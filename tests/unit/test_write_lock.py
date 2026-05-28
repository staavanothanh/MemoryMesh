"""Integration tests for the asyncio.Lock() in SqliteVecBackend.

Tests that concurrent writes are properly serialized and no deadlocks occur
when GraphStore operations use the shared backend write lock.
"""

import asyncio
import pytest

from memorymesh.memory.graph_store import GraphStore


class TestWriteLockSerialization:
    """Verify that asyncio.Lock() serializes concurrent write operations."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_serialized(self, backend):
        """10 concurrent backend.add() calls — all succeed, no transaction errors."""
        n = 10
        results = await asyncio.gather(*[
            backend.add(
                user_id="test_user",
                content=f"Concurrent memory {i}",
                embedding=[0.1] * 384,
                metadata={"importance": 3},
            )
            for i in range(n)
        ], return_exceptions=True)

        # All should succeed without exceptions
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent writes produced errors: {errors}"
        assert all(isinstance(r, str) for r in results), "All results should be memory ID strings"
        assert len(results) == n

        # Verify all were actually written
        for mid in results:
            mem = await backend._get_by_id(mid)
            assert mem is not None, f"Memory {mid} was not stored"

    @pytest.mark.asyncio
    async def test_graph_store_locked_with_backend(self, backend):
        """Concurrent graph write + memory add — no deadlock.

        GraphStore operations must respect the shared backend lock.
        Both operations use self._write_lock internally.
        """
        graph = backend.graph
        assert graph is not None, "GraphStore must be initialized"

        async def write_memory(idx: int):
            return await backend.add(
                user_id="test_user",
                content=f"Lock-test memory {idx}",
                embedding=[0.2] * 384,
            )

        async def write_graph_entity(name: str):
            return await graph.create_entity(
                name=name,
                user_id="test_user",
                entity_type="concept",
            )

        # Fire off interleaved memory adds and graph entity creates
        coros = []
        for i in range(5):
            coros.append(write_memory(i))
            coros.append(write_graph_entity(f"Entity-{i}"))

        results = await asyncio.gather(*coros, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent graph+memory operations failed: {errors}"
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_concurrent_add_and_delete(self, backend):
        """add + delete concurrently, no errors."""
        # First add a memory to delete
        mid = await backend.add(
            user_id="test_user",
            content="Memory to delete",
            embedding=[0.3] * 384,
        )

        async def adder(idx: int):
            return await backend.add(
                user_id="test_user",
                content=f"Add-during-delete {idx}",
                embedding=[0.3] * 384,
            )

        async def deleter():
            return await backend.delete(mid)

        results = await asyncio.gather(
            adder(1), adder(2), adder(3),
            deleter(),
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent add+delete failed: {errors}"
        # deleter returns True for success
        assert results[3] is True

    @pytest.mark.asyncio
    async def test_concurrent_update_and_search(self, backend):
        """update + search concurrently, no deadlock."""
        mid = await backend.add(
            user_id="test_user",
            content="Original content",
            embedding=[0.4] * 384,
            metadata={"importance": 3},
        )

        async def updater():
            return await backend.update(
                memory_id=mid,
                content="Updated content",
                metadata={"importance": 5},
            )

        async def searcher():
            return await backend.search(
                embedding=[0.4] * 384,
                user_id="test_user",
                top_k=5,
            )

        results = await asyncio.gather(
            updater(),
            searcher(),
            updater(),
            searcher(),
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent update+search failed: {errors}"
        assert results[0] is True, "First update should succeed"
        assert results[2] is True, "Second update should succeed"

    @pytest.mark.asyncio
    async def test_soft_delete_under_lock(self, backend):
        """soft_delete uses the write lock — concurrent writes safe."""
        mid = await backend.add(
            user_id="test_user",
            content="Soft-delete test",
            embedding=[0.5] * 384,
        )

        async def soft_del():
            return await backend.soft_delete(mid)

        async def add_during_delete():
            return await backend.add(
                user_id="test_user",
                content="Added during soft delete",
                embedding=[0.5] * 384,
            )

        results = await asyncio.gather(
            soft_del(),
            add_during_delete(),
            add_during_delete(),
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent soft-delete+add failed: {errors}"
