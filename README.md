# MemoryMesh

Long-term memory MCP server for AI agents. Local-first, hybrid search (ChromaDB + FTS5),
cross-session recall via 15 MCP tools.

## Quick Start

```bash
# Install
pip install -e .

# Run (stdio transport, default)
python -m memorymesh

# Run with env overrides
ROUTER_URL=http://localhost:11434/v1 \
  DEFAULT_MODEL=deepseek-v4-flash \
  python -m memorymesh

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
User -> OpenCode CLI -> LLM (deepseek-v4-flash via 9Router)
                         |
                         +-> recall(query)  -> MemoryMesh MCP Server
                         |                      +-> ChromaDB (vector)
                         |                      +-> FTS5 (keyword)
                         |                      +-> RRF fusion
                         |
                         +-> save_context_pair -> atomic fact extraction
```

**Key design choices:**
- Session context starts empty; LLM calls `recall` on demand (dynamic recall)
- Background tasks (enrichment, consolidation, fact extraction) are rate-limited
- Memories are soft-deleted (archived), not permanently removed
- Session-level memories auto-expire after 7 days

## 15 MCP Tools

| Tool | Purpose |
|------|---------|
| `remember` | Save a memory with content, tags, importance |
| `recall` | Retrieve top relevant memories by semantic query |
| `forget` | Soft-delete (archive) a memory |
| `archive_memory` | Move a memory to archive |
| `unarchive_memory` | Restore an archived memory |
| `list_memories` | List non-archived memories (paginated) |
| `ping` | Health check: memory_count + fts_connected |
| `save_system_prompt` | Save system prompt to current session |
| `save_context_pair` | Save conversation exchange, trigger fact extraction |
| `list_sessions` | List past sessions |
| `get_session_context` | View session context log |
| `new_session` | Create a fresh session (closes current) |
| `end_session` | End session (compaction + buffer flush) |
| `save_workspace_context` | Snapshot workspace state |
| `resume_session` | Restore context from a past session |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_URL` | `http://127.0.0.1:20128/v1` | 9Router endpoint |
| `DEFAULT_MODEL` | `deepseek-v4-flash` | Primary LLM model |
| `FALLBACK_MODEL` | `deepseek-v4-pro` | Fallback LLM model |
| `CHROMA_DB_PATH` | `./db/chroma` | Vector store path |
| `FTS_DB_PATH` | `./db/memory_fts.db` | Full-text search path |
| `SESSION_DB_PATH` | `./db/sessions.db` | Session store path |
| `DEFAULT_USER_ID` | `Shinn` | Default user identifier |
| `SESSION_MEMORY_TTL_DAYS` | `7` | Session memory expiry |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `sse` |
| `MCP_PORT` | `8090` | SSE transport port |

## Development

```bash
make install      # pip install -e .
make test         # run all tests
make run          # start server
make clean        # remove temp files
```

## Project Structure

```
src/memorymesh/
  config.py          App configuration (dataclasses + env)
  router.py          9Router client (retry + circuit breaker)
  embedder.py        SentenceTransformer (async thread pool)
  memory/
    manager.py       Core CRUD + scoring + background tasks
    chroma_impl.py   ChromaDB vector backend
    fts_backend.py   SQLite FTS5 backend
    hybrid_backend.py Hybrid search orchestrator
    consolidation.py Clustering + merge + TTL expiry
    fact_extractor.py Atomic fact extraction (single + batch)
    instinct.py      Pattern learning engine
    session_store.py Session lifecycle
  mcp_server/
    server.py        MCP server lifecycle
    handlers.py      Tool handler implementations
    tools.py         Tool schema definitions
tests/
  120+ tests (pytest, asyncio_mode=auto)
```

## License

MIT
