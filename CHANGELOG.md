# Changelog

All notable changes to MemoryMesh are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-05-28

### Added
- **`serve` CLI subcommand** — run MemoryMesh as an SSE HTTP server without MCP client (`memorymesh serve`)
- **`NoneEmbeddingProvider`** — zero-dependency embedding mode (`EMBEDDING_MODE=none`) for Docker deployments and air-gapped environments where embeddings are never needed
- **Docker support** — multi-stage Dockerfile with `ENTRYPOINT ["memorymesh", "serve"]`, docker-compose with `EMBEDDING_MODE=none`
- **Cross-platform file locking** — `msvcrt` (Windows) / `fcntl` (POSIX) PID lock replaces fragile `.pid` kill approach; no `portalocker` dependency
- **Embedding dimension safety** — `ensure_vector_dimension()` validates stored vs. configured dimension; never auto-drops `vec_memories`; raises `ValueError` on mismatch; dimension stored in `_metadata` table
- **Keyset cursor pagination** — `recall` now returns a cursor for paginated results; `search_with_fallback` returns 4-tuple `(results, tier, meta, next_cursor)`; cursor encodes query hash for safety
- **`extra="forbid"` on all 24 Pydantic input models** — prevents silent field drops (especially `user_id` in multi-user SSE mode)
- **Session-per-connection isolation** — `_session_map` keyed by MCP connection `id()` replaces fragile `ContextVar` (ContextVar values are invisible across asyncio tasks)
- **81 new tests** — concurrent write lock serialization, Pydantic user_id preservation (all 25 models), embedding dimension mismatch detection, session isolation, recall cursor, Docker build validation

### Fixed
- **SQLite write lock serialization** — `asyncio.Lock()` wraps all writes in `SqliteVecBackend`; `BEGIN IMMEDIATE` used everywhere to prevent "cannot start a transaction within a transaction" errors
- **`ensure_vector_dimension` deadlock** — `asyncio.Lock()` is not reentrant; callers must NOT hold `_write_lock` before calling; `TransactionContext` used inside the method to manage implicit autocommit state
- **Graph store FTS regex bug** — `search_entities_fts` regex `r"[\w\s]"` → `r"[\w\s]+"`; the old pattern fragmented FTS queries into individual characters, returning garbage results
- **Redundant Semaphore removed** — `MemoryManager._write_lock` (Semaphore) removed; backend lock is universal for all write paths (including background enrichment/consolidation)
- **Instance-level session map** — `ToolHandlers._session_map` + `asyncio.Lock()` replaces `ContextVar`; MCP's client connection `id()` keys the map; connection teardown cleans up automatically
- **Test mock warnings fixed** — 2/3 pre-existing `AsyncMock._execute_mock_call` warnings resolved in `test_codebase_adapter.py`
- **Dead code removed** — line 641 `if mem_tokens <= 0: continue` genuinely unreachable (`str({})` → `"{}"` always ≥ 1 token from tiktoken)
- **Token budget overshoot fixed** — metadata token count now deducted from available budget: `available = budget - total_tokens - meta_tokens`

### Changed
- **`search_with_fallback` signature** — now accepts optional `cursor` parameter; returns 4-tuple instead of 3-tuple (all 19 call sites updated)
- **`EmbeddingProvider.get_dimension()` ABC method** — all providers must implement; `get_embedding_dimension()` public API on embedder
- **Server init order** — embedder initialized first (gets dimension), then `ensure_vector_dimension(dim)` called before backend is fully active
- **`_current_session_id` → `await get_current_session_id()`** — 5 test failure fixes in `test_mcp_server.py`
- **Coverage all key modules ≥80%** — `manager.py` 71% → 100%, `graph_store.py` 73% → 89%, `sqlite_vec_backend.py` 73% → 81%, `config.py` 70% → 100%, `providers.py` 65% → 100%, `embedder.py` 85% → 94%

### Removed
- `_client_name_var` (ContextVar) — replaced by `_session_map` dict
- `MemoryManager._write_lock` (Semaphore) — backend `asyncio.Lock` handles all serialization
- Dead code line 641 in `manager.py` — unreachable `if mem_tokens <= 0: continue`

### Security
- `extra="forbid"` on all input models prevents silent `user_id` field drops (multi-user isolation)
- Cross-platform file lock instead of PID kill (no risk of killing wrong process)

## [0.5.0] — 2025-11-20

### Added
- Major upgrade: Knowledge Graph, Instinct v2, Lossless History, Lifecycle Automation, Dynamic Context, Embedding Factory, DX tools
- 24 MCP tools across 6 categories
- SSE transport support (MCP HTTP mode)
- Temporal decay scoring for memories
- Multi-hop GraphRAG with recursive CTE traversal
- N-gram behavioral learning (Instinct v2)
- Action Choke Point mechanism
- 3-tier hybrid search (ANN → FTS5 → chronological)
- Lossless raw history with zlib compression
- Optimistic hydration for sub-5ms context restoration

### Changed
- Input validation, path sanitization, rate limiting
- Background I/O parallelization
- Database path unified across all stores
- Vietnamese → English translation in all tool files

## [0.4.0] — 2025-10-15

### Added
- Initial MCP server implementation with 9 tools
- SQLite vector storage via sqlite-vec
- Session management and lifecycle
- Basic CLI (init, stats, sessions)
- ContextVar-based session isolation
