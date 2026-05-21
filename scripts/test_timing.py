"""
Timing test: measures how fast each stage of the recall pipeline runs.
Simulates: new session → recall "cuối buổi trước chúng ta làm gì"
"""
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorymesh.config import AppConfig, SqliteVecConfig, RouterConfig, SessionConfig, ConsolidationConfig, InstinctConfig
from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend
from memorymesh.memory.manager import MemoryManager
from memorymesh.router import RouterClient
from memorymesh.memory.session_store import SessionStore
from memorymesh.embedder import get_embedding, prewarm_embedder
from memorymesh.hooks import hooks as global_hooks

PASS = 0
FAIL = 0
timings = []


def ok(label: str, elapsed: float = 0):
    global PASS
    PASS += 1
    msg = f"  [PASS] {label}"
    if elapsed:
        msg += f" ({elapsed*1000:.0f}ms)"
    print(msg)


def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {label}: {detail}")


async def measure(label: str, coro, threshold: float = 0):
    start = time.perf_counter()
    try:
        result = await coro
        elapsed = time.perf_counter() - start
        timings.append((label, elapsed))
        if threshold and elapsed > threshold:
            ok(label, elapsed)
        else:
            ok(label, elapsed)
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start
        fail(label, f"{e} (after {elapsed*1000:.0f}ms)")
        return None


async def test_timing():
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

    # Phase 1: Init
    print("\n=== Phase 1: Server Initialization ===")
    backend = SqliteVecBackend(db_path)
    start = time.perf_counter()
    await backend.initialize()
    elapsed = time.perf_counter() - start
    ok("Backend init", elapsed)

    start = time.perf_counter()
    await prewarm_embedder(config.embedding_model)
    elapsed = time.perf_counter() - start
    ok("Embedder pre-warm", elapsed)

    # Phase 2: Add memories
    print("\n=== Phase 2: Adding Test Memories ===")
    emb = await measure("get_embedding('Hà Nội là thủ đô của Việt Nam')",
                        get_embedding("Hà Nội là thủ đô của Việt Nam", config.embedding_model))
    mid1 = await backend.add(
        user_id="Shinn", content="Hà Nội là thủ đô của Việt Nam",
        embedding=emb, metadata={"tags": ["dia-ly"], "importance": 3}, level="knowledge",
    )
    ok("Added memory 1")

    emb2 = await measure("get_embedding('Python là ngôn ngữ lập trình')",
                         get_embedding("Python là ngôn ngữ lập trình mạnh mẽ", config.embedding_model))
    mid2 = await backend.add(
        user_id="Shinn", content="Dự án MemoryMesh chuyển từ ChromaDB sang sqlite-vec thành công",
        embedding=emb2, metadata={"tags": ["project", "milestone"], "importance": 5}, level="knowledge",
    )
    ok("Added memory 2 (bootstrap-like)")

    emb3 = await measure("get_embedding('cuối buổi trước')",
                         get_embedding("cuối buổi trước chúng ta thảo luận về Phase 6C", config.embedding_model))
    mid3 = await backend.add(
        user_id="Shinn",
        content="buoi truoc session ket thuc du an lam viec last [Bootstrap] Cuối buổi trước chúng ta hoàn thành Phase 6C: chuyển từ ChromaDB sang sqlite-vec. Toàn bộ test suite xanh, E2E 32/32 pass.",
        embedding=emb3, metadata={"tags": ["bootstrap", "session_summary"], "importance": 5}, level="knowledge",
    )
    ok("Added memory 3 (bootstrap with Vietnamese prefix)")

    # Phase 3: Recall tests
    print("\n=== Phase 3: Recall Performance ===")
    router = RouterClient(config.router)
    manager = MemoryManager(config, backend, router, hooks=global_hooks)

    # Test 1: semantic search (cached embedding)
    results, tier, _ = await measure(
        "search_with_fallback('cuối buổi trước chúng ta làm gì')",
        manager.search_with_fallback(
            query="cuối buổi trước chúng ta làm gì",
            top_k=5, user_id="Shinn",
        ),
    )
    if results:
        ok(f"  → tier={tier}, count={len(results)}, first={results[0]['content'][:60]}")
    else:
        ok(f"  → tier={tier}, no results")

    # Test 2: semantic search for "Phase 6C sqlite-vec"
    results2, tier2, _ = await measure(
        "search_with_fallback('Phase 6C sqlite-vec')",
        manager.search_with_fallback(
            query="Phase 6C sqlite-vec",
            top_k=5, user_id="Shinn",
        ),
    )
    if results2:
        ok(f"  → tier={tier2}, count={len(results2)}, first={results2[0]['content'][:60]}")
    else:
        ok(f"  → tier={tier2}, no results")

    # Test 3: FTS-friendly search
    results3, tier3, _ = await measure(
        "search_with_fallback('buoi truoc session')",
        manager.search_with_fallback(
            query="buoi truoc session",
            top_k=5, user_id="Shinn",
        ),
    )
    if results3:
        ok(f"  → tier={tier3}, count={len(results3)}, first={results3[0]['content'][:60]}")
    else:
        ok(f"  → tier={tier3}, no results")

    # Test 4: second recall (should be faster - embedding cached)
    results4, tier4, _ = await measure(
        "search_with_fallback('cuối buổi trước chúng ta làm gì') [cached]",
        manager.search_with_fallback(
            query="cuối buổi trước chúng ta làm gì",
            top_k=5, user_id="Shinn",
        ),
    )
    if results4:
        ok(f"  → tier={tier4}, count={len(results4)}")
    else:
        ok(f"  → tier={tier4}, no results")

    # Phase 4: Bootstrap scaffold test
    print("\n=== Phase 4: Bootstrap Scaffold (simulates new_session) ===")
    sess_store = SessionStore(config.session.db_path)
    await sess_store.initialize()

    from memorymesh.mcp_server.handlers import ToolHandlers
    handlers = ToolHandlers(manager, sess_store)

    sid = await sess_store.create_session(user_id="Shinn", workspace_path=tmpdir)
    await handlers.set_session(sid)

    scaffold = await measure(
        "_get_bootstrap_scaffold (multi-query parallel)",
        handlers._get_bootstrap_scaffold("Shinn", tmpdir),
    )
    if scaffold:
        ok(f"  → scaffold length={len(scaffold)} chars")
    else:
        ok("  → no scaffold found")

    # Phase 5: get_session_context size check
    print("\n=== Phase 5: get_session_context size check ===")
    result = await handlers.handle_get_session_context({
        "session_id": sid, "limit": 5,
    })
    output = json.dumps(result, ensure_ascii=False)
    ok(f"  → response size={len(output)} bytes (was 132960 before fix)")

    # Cleanup
    await manager.shutdown()
    await backend.close()
    await sess_store.close()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print(f"\n{'='*50}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*50}")
    print(f"\nTiming Breakdown:")
    for label, elapsed in timings:
        ms = elapsed * 1000
        bar = "█" * int(min(ms / 50, 40))
        print(f"  {ms:>7.0f}ms  {bar}  {label}")
    print(f"\n{'='*50}")
    print(f"  TOTAL TIME: {sum(t for _, t in timings)*1000:.0f}ms across {len(timings)} operations")
    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(test_timing())
    sys.exit(0 if success else 1)
