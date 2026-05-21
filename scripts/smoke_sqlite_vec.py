"""Quick smoke test for SqliteVecBackend."""
import asyncio
import tempfile
import os

import numpy as np
from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend


async def smoke():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    b = SqliteVecBackend(path)
    await b.initialize()

    emb = np.random.rand(384).tolist()

    mid = await b.add("user1", "hello world", emb, {"tags": ["greeting"]}, level="user")
    assert mid is not None
    print(f"  add: {mid}")

    results = await b.search(emb, "user1", top_k=5)
    assert len(results) == 1
    assert results[0]["content"] == "hello world"
    print(f"  search: score={results[0]['score']:.4f}")

    fts = await b.fts_search("hello", "user1")
    assert len(fts) == 1
    print(f"  fts_search: {len(fts)} result")

    recent = await b.list_recent("user1")
    assert len(recent) == 1
    print(f"  list_recent: {len(recent)} result")

    all_m = await b.list_all("user1")
    assert len(all_m) == 1
    print(f"  list_all: {len(all_m)} result")

    wemb = await b.get_with_embeddings("user1")
    assert len(wemb) == 1
    assert len(wemb[0]["embedding"]) == 384
    print(f"  get_with_embeddings: {len(wemb)} result")

    wemb2 = await b.get_with_embeddings_by_ids([mid])
    assert len(wemb2) == 1
    print(f"  get_with_embeddings_by_ids: {len(wemb2)} result")

    upd = await b.update_metadata(mid, {"importance": 5})
    assert upd
    check = await b._get_by_id(mid)
    assert check["metadata"]["importance"] == 5
    print(f"  update_metadata: importance=5")

    upd2 = await b.update(mid, "updated content", {"tags": ["updated"]})
    assert upd2
    fts2 = await b.fts_search("updated", "user1")
    assert len(fts2) == 1
    print(f"  update: content changed")

    deleted = await b.delete(mid)
    assert deleted
    post_del = await b.list_all("user1")
    assert len(post_del) == 0
    print(f"  delete: removed")

    mid2 = await b.add("user1", "to be archived", emb)
    archived = await b.soft_delete(mid2)
    assert archived
    post_arch = await b.list_all("user1")
    assert len(post_arch) == 0
    vec_check = await b._db.execute(
        "SELECT memory_id FROM vec_memories WHERE memory_id = ?", (mid2,)
    )
    assert await vec_check.fetchone() is None
    fts_check = await b._db.execute(
        "SELECT memory_id FROM memory_fts WHERE memory_id = ?", (mid2,)
    )
    assert await fts_check.fetchone() is None
    print(f"  soft_delete+trigger: vec+fts cleaned")

    await b.close()
    os.unlink(path)
    print()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(smoke())
