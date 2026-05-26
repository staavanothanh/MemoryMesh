<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-en-red.svg" alt="English"></a>
  <a href="README.vi.md"><img src="https://img.shields.io/badge/lang-vi-blue.svg" alt="Tiếng Việt"></a>
</p>

# MemoryMesh

<p align="center">
  <img src="https://img.shields.io/badge/MCP-Compliant-brightgreen?style=for-the-badge" alt="MCP Compliant">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Database-SQLite--vec-orange?style=for-the-badge&logo=sqlite" alt="Database">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Version-0.5.0-purple?style=for-the-badge" alt="Version 0.5.0">
</p>

**Local-first persistent memory MCP server for AI agents.** v0.5.0 brings Knowledge Graphs, Behavioral Learning, Lifecycle Automation, and Dynamic Context Management.

## Quick Start

```bash
# Lite install (remote embedding API)
pip install memorymesh
# Or with local embedding model (recommended for offline use)
pip install "memorymesh[local]"

python -m memorymesh init   # one-command setup
opencode
```

### CLI Tools

```bash
memorymesh sessions --limit 20   # list recent sessions
memorymesh stats                 # show system statistics
memorymesh init                  # initialize workspace
```

## What's New in v0.5.0

| Feature | Description |
|---------|-------------|
| **🧠 Knowledge Graph** | SQLite entities/relations with `trace_entity`, `query_graph`, `create_entity`, `create_relation`. Recursive CTE for multi-hop reasoning. XML Triplet output. Safe for Plan/Read-Only mode. |
| **📜 Lossless Raw History** | Verbatim tool call logging with zlib compression via AsyncBatchLogger. Query with `recall_raw` — no more lost context on compaction. |
| **🤖 Behavioral Learning (Instinct v2)** | RAM cache with pre-compiled regex for O(1) matching. N-gram workflow pattern extraction. Auto-apply tags when confidence > 0.8. Project-scoped instincts. |
| **⚡ Lifecycle Automation** | Compressed Context Delta (<800 tokens) on auto-recall. Lazy summarization (Orphan Recovery) for missing milestones. Idle watchdog (15 min). |
| **📄 Dynamic Context Management** | Stateless keyset (cursor) pagination. Dynamic scoring pushed to SQLite CTEs. Multi-threaded token counting. |
| **🔧 Embedding Offloading** | Lightweight core: `pip install memorymesh` (no PyTorch). Optional `[local]` for SentenceTransformer. Or use remote embedding API. |
| **🛡️ Robustness** | `_safe_task_wrapper` for all background tasks. Retry with exponential backoff. Silent failure detection & logging at `ERROR` level. |

## Architecture

```mermaid
graph TD
    LLM[Agent / LLM] -->|call_tool| MM[MemoryMesh MCP Server]

    subgraph MM[MemoryMesh Server]
        VEC[(sqlite-vec ANN)] --- FTS[(FTS5 Full-Text)]
        VEC --- G[(Knowledge Graph entities/relations)]
        RAW[(Raw History Log)] --- SESS[(Session Store)]
        INST[(Instinct RAM Cache)] --- MIDDLEWARE[Tool Execution Middleware]
    end

    MM -->|recall(query, cursor?)| LLM
    LLM -->|create_entity / create_relation| G
```

**Key design:**
- Single SQLite DB with sqlite-vec for vector search + FTS5 for keyword + Graph tables for relations
- Background tasks (enrichment, consolidation, fact extraction) are rate-limited with automatic retries
- Session-level memories auto-expire after 7 days
- All background tasks wrapped in `_safe_task_wrapper` — zero silent failures

### 22 MCP Tools

| Category | Tools |
|----------|-------|
| **Memory** | `remember`, `recall`, `forget`, `archive_memory`, `unarchive_memory`, `list_memories` |
| **Knowledge Graph** | `create_entity`, `create_relation`, `query_graph`, `trace_entity` |
| **Session Lifecycle** | `new_session`, `end_session`, `resume_session`, `delete_session`, `list_sessions`, `get_session_context`, `save_system_prompt`, `preserve_session_memories` |
| **Workspace** | `commit_milestone`, `save_workspace_context`, `save_context_pair` (deprecated) |
| **Learning** | `learn_session`, `recall_raw` |
| **Utility** | `ping` |

### Hidden Gems (Under the Hood)

MemoryMesh is packed with subtle architectural decisions designed to make the AI feel more like a human colleague:

- **Zero-Latency Context (Optimistic Hydration):** Pre-computes semantic anchors before session close. Stored in RAM Cache — AI regains context in `<5ms`.
- **3-Tier Fallback Retrieval:** 1. Semantic (sqlite-vec) → 2. FTS5 keyword → 3. Chronological scan. Zero "hallucinations from amnesia".
- **Choke Point Mechanism:** After 5+ uncommitted actions, `recall` is blocked until `commit_milestone` is called. Prevents context overflow.
- **GraphRAG via SQLite CTE:** Multi-hop relation traversal uses SQL `WITH RECURSIVE` — no external graph DB needed.
- **JIT Instinct Injection:** Tool execution middleware with sliding window (deque maxlen=5). Matches against compiled regex instincts in microseconds.
- **Hybrid Pagination:** Keyset cursor pagination with dynamic scoring pushed to SQLite — O(1) deep page performance.

## Installation

### Prerequisites
- Python **3.12+**
- An OpenAI-compatible LLM endpoint (Ollama, vLLM, OpenAI, 9Router, etc.)

### Lightweight Install (Remote Embedding)
```bash
pip install memorymesh
```
Core is ~50MB. Embedding computed via remote API (set `EMBEDDING_MODE=remote` + `REMOTE_EMBEDDING_API_URL`).

### Full Install (Local Embedding)
```bash
pip install "memorymesh[local]"
```
Includes `sentence-transformers` (~1.5GB). Embedding computed locally — fully offline.

### With CLI (Rich Tables)
```bash
pip install "memorymesh[cli]"
```

### Development
```bash
pip install -e ".[test,local,cli]"
```

### Environment

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_URL` | `http://127.0.0.1:20128/v1` | LLM endpoint |
| `DEFAULT_MODEL` | `your-model` | Primary LLM model |
| `BACKGROUND_MODEL_POOL` | — | Comma-separated list of free/cheap models for background tasks |
| `EMBEDDING_MODE` | `local` | `local` (SentenceTransformer) or `remote` (API) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model name |
| `REMOTE_EMBEDDING_API_URL` | — | Remote embedding endpoint |
| `REMOTE_EMBEDDING_API_KEY` | — | API key for remote embedding |
| `VEC_DB_PATH` | `./db/memory.db` | SQLite database path |
| `AUTO_EXTRACT_FACTS` | `true` | Set to `false` to disable automatic fact extraction |
| `DEFAULT_USER_ID` | `your_user_id` | Default user (for multi-agent isolation) |

## CLI Reference

### `memorymesh sessions`
List recent sessions with status, workspace, timestamps.

### `memorymesh stats`
Show system statistics: session count, memory count, graph entities/relations, DB sizes.

### `memorymesh init`
One-command workspace setup — creates `.env`, `db/` directory, and MCP config.

## Testing

```bash
make test          # run all tests
make test-unit     # unit tests only
make test-int      # integration tests only
```

281+ tests covering:
- Memory CRUD, search fallback, consolidation, fact extraction
- Graph entities/relations, recursive CTE, cyclic detection
- Raw history logging, batch compression
- Instinct learning, N-gram extraction, RAM cache O(1) matching
- Lifecycle automation, orphan recovery, context delta
- Keyset cursor pagination, dynamic scoring
- Embedding factory, graceful fallback

## Project Structure

```
src/memorymesh/
  cli.py                  CLI tools (sessions, stats, init)
  config.py               App configuration (dataclasses + env)
  router.py               LLM router client (retry + circuit breaker)
  embedder.py             Embedding interface (factory-based)
  embeddings/
    factory.py            EmbeddingFactory (local/remote)
    providers.py          LocalEmbeddingProvider & RemoteEmbeddingProvider
  memory/
    manager.py            Core CRUD + scoring + background tasks
    sqlite_vec_backend.py Single-DB: vector + FTS5 + metadata + graph
    graph_store.py        Knowledge Graph entities/relations + CTE
    context_manager.py    Keyset cursor pagination + dynamic scoring
    consolidation.py      Clustering + merge + TTL expiry
    fact_extractor.py     Atomic fact extraction
    instinct.py           Pattern learning engine
    instinct_store.py     Instinct DB (v1 + v2 regex-based)
    instinct_manager.py   RAM cache + pre-compiled regex + N-gram extraction
    session_store.py      Session lifecycle + raw log + workspace snapshots
    async_batch_logger.py zlib-compressed batch logging
    codebase_adapter.py   External DB read-only ACL adapter
  mcp_server/
    server.py             MCP server lifecycle + idle watchdog
    handlers/             Modular handler package (5 files)
      base.py             ToolHandlers class
      _core.py            Shared constants & helpers
      semantic_filter.py  Noise detection
      tracker.py          Conversation tracker + choke point
tools/
    tools.py              Tool schema definitions
tests/
    281+ tests (pytest, asyncio_mode=auto)
```

## Multi-Agent & Cross-Device Sync

MemoryMesh uses `DEFAULT_USER_ID` for seamless multi-agent isolation:

- **Same machine, multiple agents:** Set different `DEFAULT_USER_ID` values for OpenCode vs Cline vs Cursor — isolated memory contexts.
- **Cross-device sync:** Place `db/` on Google Drive / Dropbox. Same `DEFAULT_USER_ID` = synchronized memory across machines.

## Security Notice

> [!CAUTION]
> **CRITICAL:** MemoryMesh stores all data as **plaintext** in SQLite databases (`./db/`). Never commit these to a public repository.
> ```
> db/
> .env
> ```

## Maintenance

### Rebuild Vector Index
```bash
python scripts/rebuild_vec.py
```

## License

MIT
