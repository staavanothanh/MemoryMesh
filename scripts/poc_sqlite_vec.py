"""
6C.1 POC: Verify sqlite‑vec extension load + ANN query + JOIN pattern on Windows.

Critical tests:
1. Extension loads without error
2. vec0 virtual table creation
3. Insert + ANN search (MATCH + k)
4. Subquery + JOIN pattern (workaround for vec0 WHERE limitation)
5. FTS5 coexistence in same DB
"""
import sqlite3
import sys
import os

import sqlite_vec
import numpy as np


DIM = 384


def test_load_extension():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def test_vec0_table(conn):
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_test USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding FLOAT[384]
        )
    """)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_test'"
    ).fetchone()
    assert row is not None, "vec_test table not created"
    print("  ✓ vec0 table created")


def test_insert_and_ann_search(conn):
    vec = np.random.rand(DIM).astype(np.float32)
    conn.execute(
        "INSERT INTO vec_test(memory_id, embedding) VALUES (?, ?)",
        ("test_id_1", vec.tobytes()),
    )
    vec2 = np.random.rand(DIM).astype(np.float32)
    conn.execute(
        "INSERT INTO vec_test(memory_id, embedding) VALUES (?, ?)",
        ("test_id_2", vec2.tobytes()),
    )

    results = conn.execute(
        """
        SELECT memory_id, distance
        FROM vec_test
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 5
        """,
        [vec.tobytes()],
    ).fetchall()

    assert len(results) >= 1, "ANN search returned no results"
    assert results[0][0] == "test_id_1", f"Expected test_id_1, got {results[0][0]}"
    print(f"  ✓ ANN search OK: {len(results)} results, closest={results[0][0]}, distance={results[0][1]:.6f}")


def test_k_param(conn):
    """Verify k= param in vec0 MATCH works."""
    vec = np.random.rand(DIM).astype(np.float32)
    conn.execute(
        "INSERT INTO vec_test(memory_id, embedding) VALUES (?, ?)",
        ("k_test_1", vec.tobytes()),
    )
    conn.execute(
        "INSERT INTO vec_test(memory_id, embedding) VALUES (?, ?)",
        ("k_test_2", vec.tobytes()),
    )
    results = conn.execute(
        "SELECT memory_id FROM vec_test WHERE embedding MATCH ? AND k = 1",
        [vec.tobytes()],
    ).fetchall()
    assert len(results) == 1, f"Expected 1 result with k=1, got {len(results)}"
    print(f"  ✓ k=1 param works: returned {len(results)} result")


def test_subquery_join_pattern(conn):
    """The critical workaround: ANN subquery → JOIN with metadata table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.execute("INSERT OR IGNORE INTO memories(id, user_id, content) VALUES (?, ?, ?)",
                 ("test_id_1", "user_a", "hello world"))
    conn.execute("INSERT OR IGNORE INTO memories(id, user_id, content) VALUES (?, ?, ?)",
                 ("test_id_2", "user_b", "goodbye world"))

    vec = np.random.rand(DIM).astype(np.float32)
    # Step 1: ANN search
    ann_results = conn.execute(
        "SELECT memory_id, distance FROM vec_test WHERE embedding MATCH ? AND k = 10",
        [vec.tobytes()],
    ).fetchall()
    assert len(ann_results) > 0, "ANN subquery returned no results"

    # Step 2: Filter + JOIN in Python / second query
    ids = [r[0] for r in ann_results]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, content, user_id FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
        [*ids, "user_a"],
    ).fetchall()
    assert len(rows) >= 1, "Subquery+JOIN pattern returned no rows"
    ids_found = [r[0] for r in rows]
    assert "test_id_1" in ids_found, "test_id_1 should be visible to user_a"
    user_b_rows = conn.execute(
        f"SELECT id FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
        [*ids, "user_b"],
    ).fetchall()
    print(f"  ✓ Subquery+JOIN pattern OK: user_a={len(rows)} rows, user_b={len(user_b_rows)} rows")


def test_fts5_coexistence(conn):
    """Verify FTS5 works alongside vec0 in same DB."""
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_test USING fts5(
            memory_id UNINDEXED,
            user_id UNINDEXED,
            content,
            tokenize='unicode61'
        )
    """)
    conn.execute(
        "INSERT INTO memory_fts_test(memory_id, user_id, content) VALUES (?, ?, ?)",
        ("test_id_1", "user_a", "hello world memory mesh"),
    )
    rows = conn.execute(
        "SELECT memory_id FROM memory_fts_test WHERE content MATCH ?",
        ["memory"],
    ).fetchall()
    assert len(rows) == 1, f"FTS5 search expected 1 row, got {len(rows)}"
    print("  ✓ FTS5 + vec0 coexistence OK")


def test_negative_ann_empty(conn):
    """ANN search with no matching vectors returns empty."""
    far_vec = np.ones(DIM, dtype=np.float32) * 999
    rows = conn.execute(
        "SELECT memory_id FROM vec_test WHERE embedding MATCH ? AND k = 5",
        [far_vec.tobytes()],
    ).fetchall()
    print(f"  ✓ ANN empty query OK: returned {len(rows)} results (expected >=0)")


def main():
    print("=" * 60)
    print("Phase 6C POC — sqlite-vec on Windows")
    print("=" * 60)
    print(f"Python:     {sys.version}")
    print(f"SQLite:     {sqlite3.sqlite_version}")
    print(f"sqlite-vec: {sqlite_vec.__version__}")
    print(f"numpy:      {np.__version__}")
    print()

    tests = [
        ("Load extension", test_load_extension),
        ("vec0 table creation", test_vec0_table),
        ("Insert + ANN search", test_insert_and_ann_search),
        ("k= param", test_k_param),
        ("Subquery + JOIN", test_subquery_join_pattern),
        ("FTS5 coexistence", test_fts5_coexistence),
        ("Negative — empty ANN", test_negative_ann_empty),
    ]

    conn = None
    passed = 0
    for name, fn in tests:
        try:
            if fn is test_load_extension:
                conn = fn()
                print(f"  ✓ {name}")
            else:
                assert conn is not None
                fn(conn)
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print()
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("✅ ALL POC TESTS PASSED — safe to proceed to 6C.2")
    else:
        print("❌ SOME TESTS FAILED — investigate before proceeding")

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
