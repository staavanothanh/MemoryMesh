# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev mode with all extras)
pip install -e ".[test,local,cli]"

# Run tests
make test              # all tests
make test-unit         # unit tests only
make test-int          # integration tests only

# Run a single test file
python -m pytest tests/unit/test_router.py -v

# Run a single test
python -m pytest tests/unit/test_router.py::test_something -v

# Lint & typecheck
make lint              # ruff check src/ tests/
make typecheck         # pyright src/

# Run server
python -m memorymesh            # stdio mode (default)
memorymesh serve --port 8090    # SSE HTTP mode

# CLI queries
memorymesh stats                # memory/session/instinct counts + DB sizes
memorymesh sessions --limit 20  # recent sessions

# Init workspace
python -m memorymesh init

# Clean DBs, caches, build artifacts
make clean
```

## Project Structure

```
src/memorymesh/
├── mcp_server/              # MCP protocol layer
│   ├── server.py            # MemoryMeshServer: lifecycle, tool dispatch, middleware
│   ├── tools.py             # 24 MCP tool definitions (auto-schema from Pydantic)
│   └── handlers/            # Tool implementations, trackers, semantic filters
├── memory/                  # Storage engines
│   ├── sqlite_vec_backend.py   # Main DB: vector ANN (sqlite-vec) + FTS5 + schema
│   ├── session_store.py        # Session lifecycle (create, end, resume, compact)
│   ├── graph_store.py          # Knowledge graph (entities, relations, recursive CTE)
│   ├── manager.py              # MemoryManager: orchestrates backend + consolidation + instincts
│   ├── consolidation.py        # Dedup/cluster similar memories via LLM
│   ├── instinct_store.py       # Instinct v1/v2 DB persistence
│   ├── instinct_manager.py     # Instinct lifecycle + background learning daemon
│   ├── instinct.py             # InstinctEngine: N-gram pattern extraction, scoring
│   ├── fact_extractor.py       # LLM-based fact extraction from memories
│   ├── context_manager.py      # Keyset-cursor pagination, scoring CTEs
│   ├── codebase_adapter.py     # Codebase scanning for workspace context
│   └── async_batch_logger.py   # zlib-compressed raw tool-call logging
├── utils/
│   ├── tool_middleware.py      # Sliding window + JIT instinct injection
│   ├── json_parser.py          # LLM JSON response cleanup
│   ├── rate_limiter.py         # Token-bucket per-user rate limiting
│   ├── path_sanitizer.py       # Path traversal prevention
│   ├── tokenization.py         # Tiktoken-based token counting
│   └── tool_middleware.py      # Tool execution middleware
├── embeddings/                 # Embedding provider factory
├── schemas.py                  # Pydantic models (tool inputs, domain types)
├── config.py                   # 7 config dataclasses, loaded from .env
├── router.py                   # Async LLM client with circuit breaker + fallback pool
├── embedder.py                 # Embedding engine (local / remote / none)
├── prompts.py                  # LLM prompt templates
├── cli.py                      # CLI parser (sessions, stats, init, serve)
├── hooks.py                    # Hook registry
├── errors.py                   # Custom exceptions
└── logging_.py                 # Logging setup

tests/
├── conftest.py              # Fixtures: temp DBs, backend, memory_manager, router_client
├── unit/                    # ~52 test files covering all modules
└── integration/             # E2E workflow, MCP server, Docker build, recall cursor
```

## Architecture

**MemoryMesh** is a local-first MCP memory server for AI agents. It uses a **single SQLite DB** with `sqlite-vec` (vector ANN), `FTS5` (keyword), and custom tables (knowledge graph, sessions, raw logs, instincts).

Processing pipeline per tool call:
1. `MemoryMeshServer.call_tool()` — dispatches by name → Pydantic validation → rate limiting → session auto-init → handler
2. Handler executes business logic against `MemoryManager` (backend, graph, instincts)
3. Post-call: raw log + context auto-save + instinct middleware (sliding window) + background consolidation/learning

Key subsystems:
- **3-tier hybrid search**: vector ANN → FTS5 keyword → chronological fallback, level-weighted scoring with workspace penalty
- **Action choke point**: `recall` blocked after 5 uncommitted tool calls until `commit_milestone`
- **Instinct v2**: N-gram pattern extraction from tool-call sequences, RAM-cached regex, reinforcement scoring
- **Knowledge graph**: entities + relations with recursive CTE traversal, cycle detection
- **Optimistic hydration**: semantic anchors pre-computed before session close for sub-5ms cold-start recovery

## Config

All config via environment variables (`.env` file). See `config.py` for the 7 config dataclasses:

| Config | Key Env Vars |
|--------|-------------|
| `RouterConfig` | `ROUTER_URL`, `DEFAULT_MODEL`, `FALLBACK_MODEL`, `BACKGROUND_MODEL_POOL` |
| `SqliteVecConfig` | `VEC_DB_PATH`, `VEC_AUTO_MIGRATE` |
| `SessionConfig` | `SESSION_DB_PATH`, `SESSION_AUTO_*` flags |
| `ConsolidationConfig` | `CONSOLIDATION_SIMILARITY`, `CONSOLIDATION_INTERVAL` |
| `InstinctConfig` | `INSTINCT_DB_PATH`, `INSTINCT_V2_MAX`, `INSTINCT_CONFIDENCE_FLOOR` |
| `EmbeddingConfig` | `EMBEDDING_MODE` (local/remote/none), `EMBEDDING_MODEL` |

## Key Design Decisions

- **Pydantic for input validation**: every MCP tool input is a `BaseToolInput` subclass with `extra="forbid"` — unknown fields rejected
- **Asyncio throughout**: all DB/network operations async; semaphore-limited LLM calls (max 3 concurrent)
- **Fetch-and-release CLI**: CLI commands open DB → query → close → render (no long-lived connections)
- **Embedding modes**: `local` (sentence-transformers, ~1.5GB), `remote` (API), `none` (zero-dependency, Docker)
- **MCP transports**: stdio (default for embedded agents) and SSE (Docker/remote)
- **Test isolation**: each test gets temp DB files; `_force_unlink` with retries for Windows PermissionError
