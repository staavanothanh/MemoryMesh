"""
End-to-end integration test for MemoryMesh with SqliteVecBackend.

Tests: backend init, CRUD, semantic search, FTS fallback, session ops, archive/restore.
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorymesh.config import AppConfig, SqliteVecConfig, RouterConfig, SessionConfig, ConsolidationConfig, InstinctConfig
from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend
from memorymesh.memory.manager import MemoryManager
from memorymesh.router import RouterClient
from memorymesh.memory.session_store import SessionStore
from memorymesh.hooks import hooks as global_hooks
from memorymesh.embedder import get_embedding
from memorymesh.schemas import SearchResult

PASS = 0
FAIL = 0


def ok(label: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {label}")


def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {label}: {detail}")


async def test_e2e():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "memory.db")
    sess_path = os.path.join(tmpdir, "sessions.db")

    config = AppConfig(
        router=RouterConfig(url="http://127.0.0.1:20128/v1", timeout_s=5),
        sqlite_vec=SqliteVecConfig(db_path=db_path),
        session=SessionConfig(db_path=sess_path, auto_create_session=False),
        consolidation=ConsolidationConfig(enabled=False),
        instinct=InstinctConfig(enabled=False),
    )
    config.validate()

    backend = SqliteVecBackend(db_path)
    await backend.initialize()
    ok("Backend initialized")

    # 1. Add memory
    emb = await get_embedding("Hà Nội là thủ đô của Việt Nam", config.embedding_model)
    mid1 = await backend.add(
        user_id="Shinn",
        content="Hà Nội là thủ đô của Việt Nam",
        embedding=emb,
        metadata={"tags": ["dia-ly"], "importance": 3},
        level="knowledge",
    )
    assert mid1 and isinstance(mid1, str)
    ok(f"Added memory: {mid1[:12]}")

    # 2. Add more memories
    emb2 = await get_embedding("Python là ngôn ngữ lập trình mạnh mẽ", config.embedding_model)
    mid2 = await backend.add(user_id="Shinn", content="Python là ngôn ngữ lập trình mạnh mẽ", embedding=emb2, level="user")

    emb3 = await get_embedding("Dự án MemoryMesh sử dụng sqlite-vec", config.embedding_model)
    mid3 = await backend.add(user_id="Shinn", content="Dự án MemoryMesh sử dụng sqlite-vec", embedding=emb3, level="session")

    ok("Added 3 memories total")

    # 3. Semantic search
    q_emb = await get_embedding("thủ đô việt nam", config.embedding_model)
    results = await backend.search(q_emb, "Shinn", top_k=5)
    assert len(results) >= 1
    assert any("Hà Nội" in r["content"] for r in results)
    ok("Semantic search returns relevant result")

    # 4. Search with level filter
    session_only = await backend.search(q_emb, "Shinn", top_k=5, level_filter=["session"])
    assert len(session_only) >= 1
    assert all(r["metadata"]["level"] == "session" for r in session_only)
    ok("Level filter works")

    # 5. FTS search
    fts_results = await backend.fts_search("thủ đô", "Shinn", limit=10)
    assert len(fts_results) >= 1
    assert any("Hà Nội" in r["content"] for r in fts_results)
    ok("FTS search returns result")

    fts_no_match = await backend.fts_search("zzzzzzzz", "Shinn", limit=10)
    assert len(fts_no_match) == 0
    ok("FTS search empty for no match")

    # 6. List all
    all_mem = await backend.list_all("Shinn", limit=10)
    assert len(all_mem) == 3
    ok("list_all returns all 3 memories")

    # 7. List with pagination
    page1 = await backend.list_all("Shinn", limit=2, offset=0)
    page2 = await backend.list_all("Shinn", limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 1
    ok("Pagination works")

    # 8. Get with embeddings
    with_emb = await backend.get_with_embeddings("Shinn", limit=10)
    assert len(with_emb) == 3
    assert all(len(e.get("embedding", [])) > 0 for e in with_emb)
    ok("get_with_embeddings returns 3 items with embeddings")

    # 9. Get by IDs
    by_ids = await backend.get_with_embeddings_by_ids([mid1, mid2])
    assert len(by_ids) == 2
    ok(f"get_with_embeddings_by_ids returns {len(by_ids)} items")

    # 10. Update metadata
    updated = await backend.update_metadata(mid1, {"importance": 5, "summary": "Thủ đô"})
    assert updated is True
    updated_mem = await backend._get_by_id(mid1)
    assert updated_mem["metadata"]["importance"] == 5
    ok("update_metadata works")

    # 11. Update content + metadata
    updated = await backend.update(mid2, "Python là ngôn ngữ lập trình tuyệt vời", {"importance": 4})
    assert updated is True
    by_id = await backend._get_by_id(mid2)
    assert "tuyệt vời" in by_id["content"]
    ok("update content + metadata works")

    # 12. Soft delete
    sd = await backend.soft_delete(mid3)
    assert sd is True
    del_list = await backend.list_all("Shinn", limit=10)
    assert len(del_list) == 2
    ok("Soft delete removes from list_all")

    # 13. Hard delete
    deleted = await backend.delete(mid1)
    assert deleted is True
    hard_list = await backend.list_all("Shinn", limit=10)
    assert mid1 not in [m["id"] for m in hard_list]
    ok("Hard delete works")

    # 14. Delete nonexistent returns False
    not_found = await backend.delete("nonexistent-id")
    assert not_found is False
    ok("Delete nonexistent returns False")

    # 15. Update nonexistent returns False
    not_found = await backend.update("nonexistent-id", "content", {})
    assert not_found is False
    ok("Update nonexistent returns False")

    # 16. FTS list_recent
    recent = await backend.list_recent("Shinn", limit=5)
    assert len(recent) > 0
    ok("list_recent returns results")

    # 17. FTS with level filter
    user_level = await backend.fts_search("Python", "Shinn", limit=10, level_filter=["user"])
    assert all(r.get("id") == mid2 for r in user_level)
    ok("FTS with level filter works")

    # 18. list_recent with level filter
    recent_level = await backend.list_recent("Shinn", limit=10, level_filter=["user"])
    assert len(recent_level) >= 1
    ok("list_recent with level filter works")

    # ── Manager-level tests ──
    router = RouterClient(config.router)
    manager = MemoryManager(config, backend, router, hooks=global_hooks)

    # 19. search_with_fallback — fallback chain works
    results, tier, _ = await manager.search_with_fallback(query="ngôn ngữ lập trình", top_k=5, user_id="Shinn")
    assert len(results) > 0
    ok(f"search_with_fallback tier={tier}, count={len(results)}")

    # 20. search_with_fallback — empty query falls to FTS or chronological
    results, tier, _ = await manager.search_with_fallback(query="zzzzz", top_k=5, user_id="Shinn")
    assert tier in ("fts_keyword", "chronological", "empty")
    ok(f"search_with_fallback empty query tier={tier}")

    # 21. search_memory with workspace filter
    mem_results = await manager.search_memory(query="Python", top_k=5, user_id="Shinn", workspace_path="/tmp")
    ok(f"search_memory workspace filter returns {len(mem_results)} results")

    # 22. list_memories
    records = await manager.list_memories(limit=10, user_id="Shinn")
    assert len(records) >= 1
    ok(f"list_memories returns {len(records)}")

    # 23. forget_memory (archive)
    archived = await manager.forget_memory(mid2)
    assert archived is True
    after_archive = await manager.list_memories(limit=10, user_id="Shinn")
    assert mid2 not in [r.id for r in after_archive]
    ok("forget_memory archives and hides from list")

    # 24. archive_memory / unarchive_memory
    # Create a fresh memory for archive test
    emb4 = await get_embedding("archive test memory", config.embedding_model)
    mid4 = await backend.add(user_id="Shinn", content="archive test memory", embedding=emb4)
    arch1 = await manager.archive_memory(mid4)
    assert arch1 is True
    unarch = await manager.unarchive_memory(mid4)
    assert unarch is True
    ok("archive/unarchive cycle works")

    # ── Session store tests ──
    sess_store = SessionStore(config.session.db_path)
    await sess_store.initialize()

    # 25. Create session
    sid = await sess_store.create_session(user_id="Shinn", system_prompt="Test session", workspace_path=tmpdir)
    assert sid
    ok(f"Session created: {sid[:12]}")

    # 26. Get session
    sess = await sess_store.get_session(sid)
    assert sess["session_id"] == sid
    ok("get_session returns correct session")

    # 27. Log + get context
    await sess_store.log_context(sid, "user", "hello", "test")
    await sess_store.log_context(sid, "assistant", "hi there", "test")
    ctx = await sess_store.get_context_log(sid, limit=10)
    assert len(ctx) == 2
    ok(f"Context log has {len(ctx)} entries")

    # 28. End session
    await sess_store.end_session(sid)
    ok("Session ended")

    # 29. List sessions
    sessions = await sess_store.list_sessions("Shinn", limit=5)
    assert len(sessions) >= 1
    ok(f"list_sessions returns {len(sessions)} session(s)")

    # 30. Ping-style health check
    health_count = await backend.list_all("Shinn", limit=10000)
    fts_health = await backend.fts_search("health", "Shinn", limit=1)
    assert len(health_count) >= 0
    assert isinstance(fts_health, list)
    ok("Health check OK")

    # Cleanup
    await manager.shutdown()
    await backend.close()
    await sess_store.close()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"  RESULTS:  {PASS} passed, {FAIL} failed")
    print(f"{'='*50}")
    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(test_e2e())
    sys.exit(0 if success else 1)
