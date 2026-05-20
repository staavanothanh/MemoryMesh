import pytest
import pytest_asyncio
from memorymesh.memory.fts_backend import FTSBackend, sanitize_fts_query


@pytest_asyncio.fixture
async def fts(fts_config):
    backend = FTSBackend(fts_config.db_path)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_initialize_creates_table(fts):
    cursor = await fts._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
    )
    row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_add_and_search(fts):
    await fts.add("id1", "Hà Nội là thủ đô của Việt Nam", "user1")
    await fts.add("id2", "Python là ngôn ngữ lập trình", "user1")

    results = await fts.search("thủ đô", "user1", limit=10)
    assert len(results) >= 1
    assert results[0]["id"] == "id1"


@pytest.mark.asyncio
async def test_search_filters_by_user(fts):
    await fts.add("id1", "Alice's memory", "user_a")
    await fts.add("id2", "Bob's memory", "user_b")

    results_a = await fts.search("memory", "user_a", limit=10)
    results_b = await fts.search("memory", "user_b", limit=10)

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0]["id"] == "id1"
    assert results_b[0]["id"] == "id2"


@pytest.mark.asyncio
async def test_search_empty_query(fts):
    results = await fts.search("", "user1", limit=10)
    assert results == []


@pytest.mark.asyncio
async def test_search_no_match(fts):
    await fts.add("id1", "Hello world", "user1")
    results = await fts.search("zzzzz", "user1", limit=10)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete(fts):
    await fts.add("id1", "To be deleted", "user1")
    await fts.delete("id1")

    results = await fts.search("deleted", "user1", limit=10)
    ids = [r["id"] for r in results]
    assert "id1" not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent(fts):
    result = await fts.delete("non-existent")
    assert result is True


def test_sanitize_fts_query():
    assert sanitize_fts_query("Hello, world!") == "Hello world"
    assert sanitize_fts_query("  spaces   ") == "spaces"
    assert sanitize_fts_query("") == ""


@pytest.mark.asyncio
async def test_sanitize_special_chars(fts):
    await fts.add("id1", "test data", "user1")
    results = await fts.search("test!!! data***", "user1", limit=10)
    assert len(results) >= 1
