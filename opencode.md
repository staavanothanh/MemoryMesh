# MemoryMesh — Long-term Memory MCP Server

## Project Identity
Persistent memory system for AI agents. Local-first, hybrid search (sqlite-vec + FTS5),
automatic cross-session recall via MCP protocol.

Tech stack: Python 3.12+, sqlite-vec, SQLite FTS5, asyncio, sentence-transformers, MCP SDK (v1).

## Architecture

```
Session Start ──> Bootstrap ──> Inject state summary into context
                     │
User Request ───────> LLM ──────────────> If reference to past detected
                                              │
                                              └──> recall(query) ──> sqlite-vec + FTS5
                                                                       (hybrid with fallback)
                                                                       │
                                                                  Tier 1: vector ANN
                                                                  Tier 2: FTS keyword
                                                                  Tier 3: chronological

Session End ─────────> Teardown Snapshot ──> Compaction ──> Bootstrap Snapshot
```

- **Chat** goes directly from client to LLM, NOT through MemoryMesh
- **MemoryMesh** is purely a background MCP server: enriches, consolidates, retrieves
- **Bootstrap**: `end_session` creates a summary; `new_session` auto-injects it
- **Dynamic Recall**: triggered by LLM's `recall` call, guarded by token budget
- **Single-DB**: sqlite-vec for vector, FTS5 for keyword — all in one SQLite file
- **Checkpoint Model**: Call `commit_milestone` when finishing a logical block of work. Action-based choke point blocks `recall` after 5+ uncommitted actions.

## 16 MCP Tools

| Tool | Purpose |
|------|---------|
| `remember` | Save a memory with content, tags, importance, level |
| `recall` | Retrieve top relevant memories by query |
| `forget` | Soft-delete (archive) a memory by ID |
| `archive_memory` | Move a memory into archive |
| `unarchive_memory` | Restore an archived memory |
| `list_memories` | List non-archived memories (paginated) |
| `ping` | Health check |
| `save_system_prompt` | Save system prompt to current session |
| **`commit_milestone`** | **Commit a checkpoint: summary, tasks_done, next_steps. Releases hostage data.** |
| `save_context_pair` | DEPRECATED — use commit_milestone instead |
| `list_sessions` | List past sessions |
| `get_session_context` | View a session's full context log |
| `new_session` | Start a session |
| `end_session` | End session (triggers compaction + fact buffer flush) |
| `save_workspace_context` | Snapshot workspace files, git status, deps |
| `resume_session` | Restore a past session's context |

## Key Conventions

1. **Async first**: All I/O uses `asyncio`. Never block the event loop.
2. **Type hints**: All functions MUST have complete type annotations.
3. **Error handling**: Use `MemoryMeshError` hierarchy from `errors.py`. No bare `except Exception`.
4. **Background tasks**: Use `manager._create_tracked_task()` instead of raw `asyncio.create_task()`.
5. **Rate-limiting**: All expensive background ops (consolidation 60s, fact resolution 120s, expiry 60s) are rate-limited per user.
6. **Workspace isolation**: Memories tagged with `workspace_path`; hierarchical visibility.
7. **Soft delete**: `forget` and `archive` mark `deleted=True`; use `unarchive_memory` to restore. `delete_session` performs a permanent hard delete (vector + logs + session record).
8. **Memory expiry**: Session-level memories older than `session_memory_ttl_days` (default 7) are auto-expired.
9. **Fact batching**: Up to 3 conversation pairs are batched into a single LLM call.
10. **Token budget**: Recall enforces configurable `token_budget` (default 1000).
11. **Session tagging**: Memories tagged with `session:<id>` for cleanup on end_session.
12. **Action-based choke point**: 5+ uncommitted actions blocks `recall` until `commit_milestone` is called. READ_ONLY tools (recall, remember, ping, list*) do NOT count as actions.
13. **Hostage pattern**: Blocked `recall` data is cached in the tracker; calling `commit_milestone` releases it instantly (Zero-Round-Trip).
14. **Teardown hook**: Session end or shutdown flushes unsaved context as `[SESSION-FINAL-SNAPSHOT]`.
15. **Hybrid auto-snapshot**: Idle > 60s with uncommitted actions triggers an auto-snapshot (low importance safety net).
16. **Plan Mode compatible**: Plan mode never increments uncommitted_actions (no mutate tools), so choke point never triggers. Call `commit_milestone` once when plan is finalized.

## Data Storage

```
./db/
  memory.db         Vector embeddings (sqlite-vec) + FTS5 + metadata (single DB)
  sessions.db       Session store (context_log, snapshots)
  instincts.db      Pattern learning store
```

## Compatibility

**OS**: Windows, macOS, Linux.
**Python**: 3.12+.
**Clients**: Any MCP-compatible agent (OpenCode, Claude Code, Cursor, Continue.dev, Cline, etc.).

## Directory Structure

```
src/memorymesh/
  config.py          AppConfig, dataclasses, from_env()
  router.py          LLM HTTP client (retry + fallback + circuit breaker)
  embedder.py        SentenceTransformer (async thread pool, singleton)
  scanner.py         Codebase directory scanner
  hooks.py           Pub-sub event hook registry
  prompts.py         LLM system prompts
  errors.py          MemoryMeshError hierarchy

  memory/
    manager.py       Core CRUD + scoring + enrichment + background tasks
    sqlite_vec_backend.py  Single-DB: vector + FTS5 + metadata
    consolidation.py       Similarity clustering + LLM merge + TTL expiry
    fact_extractor.py      Atomic fact extraction (single + batch)
    instinct.py            Pattern learning engine
    instinct_store.py      Instinct SQLite storage
    session_store.py       Session lifecycle + context log + snapshots

  mcp_server/
    server.py        MCP server lifecycle + tool dispatch
    handlers.py      All 15 tool handler implementations
    tools.py         MCP Tool schema definitions

Makefile              Build targets (cross-platform, python-based clean)
setup.sh              macOS/Linux one-command setup
setup.ps1             Windows one-command setup
README.md             Project docs
opencode.md           Project instruction file
.github/workflows/    CI/CD workflow (GitHub Actions)
tests/                130+ tests (pytest, asyncio_mode=auto)
```

## Coding Standards

- **Language**: Code/comments in English. Use English for all identifiers.
- **Imports**: stdlib -> third-party -> local. Absolute imports preferred.
- **Naming**: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Testing**: pytest with `asyncio_mode=auto`. Use tmp dirs for DB tests. 130+ tests required.
- **Cross-platform**: Never use shell-specific commands in Python code. Use `pathlib` for paths.
- **Commits**: Conventional commits (`feat`/`fix`/`refactor`/`perf`/`test`/`docs`).
