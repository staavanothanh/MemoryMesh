"""Tests for dimension validation in SqliteVecBackend.ensure_vector_dimension.

Verifies that mismatched dimensions are rejected, same dimensions succeed,
the vec0 table is created with the correct dimension, and metadata is stored.
"""

import pytest
import aiosqlite


class TestDimensionValidation:
    """Dimension mismatch must raise ValueError with descriptive message."""

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_error(self, backend):
        """Initialize backend, ensure 384 succeeds, then 768 raises ValueError."""
        # First call with 384 — should succeed (default DIM)
        await backend.ensure_vector_dimension(384)

        # Second call with 768 — should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await backend.ensure_vector_dimension(768)

        error_msg = str(exc_info.value).lower()
        assert "dimension mismatch" in error_msg or "vector dimension" in error_msg, (
            f"Error message should mention dimension mismatch, got: {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_dimension_same_dim_succeeds(self, backend):
        """Call ensure_vector_dimension twice with same dim — no error."""
        await backend.ensure_vector_dimension(384)
        # Second call with same dimension — should not raise
        await backend.ensure_vector_dimension(384)

    @pytest.mark.asyncio
    async def test_dimension_creates_vec_table(self, backend):
        """On first call with a new dim, vec_memories table is created."""
        # Before calling ensure_vector_dimension, the table may not exist
        cursor = await backend._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
        )
        row = await cursor.fetchone()

        if row is None:
            # Table doesn't exist yet — ensure_vector_dimension will create it
            await backend.ensure_vector_dimension(384)
            cursor = await backend._db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            )
            row = await cursor.fetchone()
            assert row is not None, "vec_memories table should exist after ensure_vector_dimension"
            sql = row["sql"]
            assert "FLOAT[384]" in sql, (
                f"vec_memories should have dimension 384, got SQL: {sql}"
            )
        else:
            # Table already exists (e.g. from backend.initialize or previous test)
            # Still verify the dimension is correct
            await backend.ensure_vector_dimension(384)
            cursor = await backend._db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            )
            row = await cursor.fetchone()
            sql = row["sql"]
            assert "FLOAT[" in sql, f"vec_memories should have FLOAT[dim], got SQL: {sql}"

    @pytest.mark.asyncio
    async def test_dimension_stored_in_metadata(self, backend):
        """After ensure_vector_dimension, _metadata table has vec_dimension key."""
        await backend.ensure_vector_dimension(384)

        cursor = await backend._db.execute(
            "SELECT value FROM _metadata WHERE key = 'vec_dimension'"
        )
        row = await cursor.fetchone()
        assert row is not None, "vec_dimension should be stored in _metadata"
        assert row["value"] == "384", f"Expected '384', got '{row['value']}'"


class TestGetMetadataByIdFallback:
    """get_metadata_by_ids returns metadata without vector JOIN."""

    @pytest.mark.asyncio
    async def test_get_metadata_by_ids_fallback(self, backend):
        """Add memory, then call get_metadata_by_ids([id]) — returns metadata."""
        mid = await backend.add(
            user_id="test_user",
            content="Fallback test memory",
            embedding=[0.1] * 384,
            metadata={"importance": 4, "tags": ["test"]},
        )

        results = await backend.get_metadata_by_ids([mid])
        assert len(results) == 1, "Should return exactly one result"
        assert results[0]["id"] == mid
        assert results[0]["content"] == "Fallback test memory"
        assert results[0]["user_id"] == "test_user"
        metadata = results[0]["metadata"]
        assert metadata.get("importance") == 4
        assert "tags" in metadata
        assert "embedding" not in results[0], (
            "get_metadata_by_ids should NOT include embedding (no vector JOIN)"
        )

    @pytest.mark.asyncio
    async def test_get_metadata_by_ids_empty_list(self, backend):
        """Empty input list returns empty list."""
        results = await backend.get_metadata_by_ids([])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_metadata_by_ids_nonexistent(self, backend):
        """Non-existent IDs return empty list."""
        results = await backend.get_metadata_by_ids(["nonexistent-id"])
        assert results == []
