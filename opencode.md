# MemoryMesh — Long-term Memory MCP Server

## Project Identity
Persistent memory system for AI agents. Local-first, hybrid search (ChromaDB + FTS5),
automatic cross-session recall via MCP protocol.

Tech stack: Python 3.12+, ChromaDB, SQLite FTS5, asyncio, sentence-transformers, MCP SDK (v1).

## Architecture

```
User -> OpenCode CLI -> LLM (deepseek-v4-flash via 9Router)
                         |
                         +-> recall(query) tool  -> MemoryMesh MCP Server
                         |                          +-> chroma_impl (vector search)
                         |                          +-> fts_backend  (keyword search)
                         |                          +-> hybrid_utils (RRF fusion)
                         |
                         +-> save_context_pair -> MemoryMesh -> atomic fact extraction
```

- **Main chat** goes directly from OpenCode to 9Router, NOT through MemoryMesh
- **MemoryMesh** is purely a background MCP server: it enriches, consolidates, and retrieves memories
- **Session context starts empty** — LLM must call `recall` on demand (dynamic recall)

## 15 MCP Tools

| Tool | Purpose |
|------|---------|
| `remember` | Save a memory with content, tags, importance, level |
| `recall` | Retrieve top relevant memories by query |
| `forget` | Soft-delete (archive) a memory by ID |
| `archive_memory` | Move a memory into archive |
| `unarchive_memory` | Restore an archived memory |
| `list_memories` | List non-archived memories (paginated) |
| `ping` | Health check — returns memory_count + fts_connected |
| `save_system_prompt` | Save system prompt to current session |
| `save_context_pair` | Save user+assistant exchange, trigger fact extraction |
| `list_sessions` | List past sessions |
| `get_session_context` | View a session's full context log |
| `new_session` | Create a fresh session (closes current one) |
| `end_session` | End session (triggers compaction + fact buffer flush) |
| `save_workspace_context` | Snapshot workspace files, git status, deps |
| `resume_session` | Restore a past session's context |

## Key Conventions

1. **Async first**: All I/O uses `asyncio`. Never block the event loop. ChromaDB sync calls wrapped in `asyncio.to_thread`.
2. **Type hints**: All functions MUST have complete type annotations.
3. **Error handling**: Use `MemoryMeshError` hierarchy from `errors.py`. No bare `except Exception`.
4. **Background tasks**: Use `manager._create_tracked_task()` instead of raw `asyncio.create_task()`. Tasks are tracked in a set and cancelled on shutdown.
5. **Rate-limiting**: All expensive background ops (consolidation 60s, fact resolution 120s, expiry 60s) are rate-limited per user.
6. **Workspace isolation**: Memories tagged with `workspace_path`; hierarchical visibility (siblings visible, parent→children visible, child→parent invisible).
7. **Soft delete**: `forget` and `archive` mark `archived=True`; archived memories excluded from recall/list. Use `unarchive_memory` to restore.
8. **Memory expiry**: Session-level memories older than `session_memory_ttl_days` (default 7) are auto-expired.
9. **List/recall filters**: Consolidated, expired, and archived memories are filtered out automatically.
10. **Fact batching**: Up to 3 conversation pairs are batched into a single LLM call for atomic fact extraction.

## Data Storage

All persistent data lives under `.opencode/data/` (per-workspace sandbox):
```
.opencode/data/
  chroma/            Vector embeddings (ChromaDB PersistentClient)
  memory_fts.db      Full-text search (SQLite FTS5)
  sessions.db        Session store (context_log, snapshots)
  instincts.db       Pattern learning store
```

Legacy data under `./db/` is still supported via env vars.

## Directory Structure

```
src/memorymesh/
  config.py          AppConfig, dataclasses, from_env()
  router.py          9Router HTTP client (retry + fallback + circuit breaker)
  embedder.py        SentenceTransformer (async thread pool, singleton)
  scanner.py         Codebase directory scanner
  hooks.py           Pub-sub event hook registry
  prompts.py         LLM system prompts
  errors.py          MemoryMeshError hierarchy

  memory/
    manager.py       Core CRUD + scoring + enrichment + background tasks
    chroma_impl.py   ChromaDB backend (async + asyncio.to_thread)
    fts_backend.py   SQLite FTS5 backend
    hybrid_backend.py Hybrid search (ChromaDB + FTS5 + RRF fusion)
    hybrid_utils.py  RRF fusion algorithm
    consolidation.py Similarity clustering + LLM merge + TTL expiry
    fact_extractor.py Atomic fact extraction (single + batch)
    instinct.py      Pattern learning engine
    instinct_store.py Instinct SQLite storage
    session_store.py Session lifecycle + context log + snapshots

  mcp_server/
    server.py        MCP server lifecycle + tool dispatch
    handlers.py      All 15 tool handler implementations
    tools.py         MCP Tool schema definitions

Makefile              Build targets (install, test, run, clean)
README.md             Project docs (quick start, architecture, 15 tools)
opencode.md           Project instruction file (auto-injected by OpenCode)
.github/workflows/    CI/CD workflow (GitHub Actions)
tests/                148 tests (pytest, asyncio_mode=auto)
```

## Coding Standards

- **Language**: Code/comments in English. Prompt descriptions in Vietnamese (end-user facing).
- **Imports**: stdlib -> third-party -> local. Absolute imports preferred.
- **Naming**: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Testing**: pytest with `asyncio_mode=auto`. Use tmp dirs for DB tests. 148+ tests required.
- **Commits**: Conventional commits (`feat`/`fix`/`refactor`/`perf`/`test`/`docs`).
