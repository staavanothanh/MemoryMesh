# MemoryMesh

**Local-first persistent memory MCP server for AI agents.**
Hybrid search (vector + FTS5) in a single SQLite database, cross-session recall via MCP tools.

## Quick Start

```bash
# Clone & enter
git clone https://github.com/staavanothanh/MemoryMesh.git
cd MemoryMesh

# Create venv & install
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -e ".[test]"         # install with test deps
python -m memorymesh init        # one-command setup (creates .env, MCP config, instructions)

# Start OpenCode — MCP server auto-launches
opencode
```

Or step by step:

```bash
pip install -e ".[test]"
cp .env.example .env             # configure your LLM endpoint
python -m memorymesh             # start standalone MCP server
```

> **Note:** On the very first run, MemoryMesh will automatically download the `sentence-transformers` embedding model (~100–300 MB) to your local cache. This may take 1–2 minutes depending on your network connection.

## Architecture

```
User -> Any MCP Client -> LLM (via your router)
                           |
                           +-> recall(query)  -> MemoryMesh MCP Server
                           |                      +-> sqlite-vec (vector ANN)
                           |                      +-> FTS5 (keyword)
                           |                      +-> RRF fusion
                           |
                           +-> save_context_pair -> atomic fact extraction
```

**Key design:**
- Session context starts empty; LLM calls `recall` on demand (dynamic recall)
- Single SQLite DB with sqlite-vec for vector search + FTS5 for keyword
- Background tasks (enrichment, consolidation, fact extraction) are rate-limited
- Session-level memories auto-expire after 7 days

## Cost Management & Dual-LLM Architecture

MemoryMesh uses **two independent LLM layers** to separate interactive performance from background cost:

| Layer | Variable | Typical Models | Purpose |
|-------|----------|----------------|---------|
| **Foreground (Chat)** | `DEFAULT_MODEL` / `FALLBACK_MODEL` | GPT-5.5, Claude 4.7 Opus, Gemini 3.5 Flash, DeepSeek V4-Pro | Direct user interaction via the MCP client |
| **Background (Data)** | `BACKGROUND_MODEL_POOL` | Gemini 2.5 Flash, Llama 3.1 8B, DeepSeek V4 Flash | Bootstrap snapshots, atomic fact extraction, session compaction |

### How it works
- Your **primary chat model** handles all user-facing reasoning — choose the best model you can afford.
- **Background tasks** (fact extraction, snapshot generation, compaction) run on a separate pool of *free or low-cost* models defined in `BACKGROUND_MODEL_POOL`. The pool uses a **cascade strategy**: if the first model fails, it automatically tries the next in the list.
- If `BACKGROUND_MODEL_POOL` is empty, background tasks fall back to your primary chat model.

### Economy Mode
Set `AUTO_EXTRACT_FACTS=false` in your `.env` to disable all automatic atomic fact extraction entirely. This eliminates background LLM calls (zero token cost) while keeping narrative-thread storage and cross-session recall fully operational.

### Recommended background models
These models deliver excellent results for summarization and extraction at near-zero cost:

```env
BACKGROUND_MODEL_POOL=openrouter/google/gemini-2.5-flash,openrouter/deepseek/deepseek-v4-flash,openrouter/qwen/qwen-2.5-7b-instruct,openrouter/meta-llama/llama-3.1-8b-instruct
```

## Setup

### Prerequisites
- Python **3.12+**
- An OpenAI-compatible LLM endpoint (Ollama, vLLM, OpenAI, 9Router, etc.)

### Environment

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_URL` | `http://127.0.0.1:20128/v1` | LLM endpoint |
| `DEFAULT_MODEL` | `your-model` | Primary LLM model |
| `BACKGROUND_MODEL_POOL` | — | Comma-separated list of free/cheap models for background tasks |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `VEC_DB_PATH` | `./db/memory.db` | SQLite database path |
| `AUTO_EXTRACT_FACTS` | `true` | Set to `false` to disable automatic fact extraction (Economy Mode) |
| `DEFAULT_USER_ID` | `your_user_id` | Default user |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `sse` |
| `MCP_PORT` | `8090` | SSE port |

### Using with CLI Agents

MemoryMesh is an **MCP server** — compatible with any MCP client:

| Agent | Setup |
|-------|-------|
| **OpenCode** | `python -m memorymesh init` generates `.opencode/opencode.json` + instruction file automatically. Then just run `opencode`. |
| **Claude Code** | Add MCP server in Claude Code config |
| **Cursor** | Add MCP server in Cursor settings |
| **Continue.dev** | Add MCP server in `~/.continue/config.json` |
| **Cline / Roo Code** | Add MCP server in VS Code extension settings |
| **Any MCP client** | `python -m memorymesh` (stdio) or `http://localhost:8090` (SSE) |

### Docker

```bash
docker compose -f docker/docker-compose.yml up -d
```

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

## Development

```bash
make install      # pip install -e ".[test]"
make test         # run all tests
make run          # start server
make clean        # remove temp files
```

## Project Structure

```
src/memorymesh/
  config.py         App configuration (dataclasses + env)
  router.py         LLM router client (retry + circuit breaker)
  embedder.py       SentenceTransformer (async thread pool)
  memory/
    manager.py      Core CRUD + scoring + background tasks
    sqlite_vec_backend.py  Single-DB: vector + FTS5 + metadata
    consolidation.py       Clustering + merge + TTL expiry
    fact_extractor.py      Atomic fact extraction
    instinct.py            Pattern learning engine
    session_store.py       Session lifecycle
  mcp_server/
    server.py        MCP server lifecycle
    handlers.py      Tool handler implementations
    tools.py         Tool schema definitions
tests/
   130+ tests (pytest, asyncio_mode=auto)
```

## Security Notice

> **CRITICAL:** MemoryMesh is a local-first system. All conversation logs, project contexts, and extracted memories are stored as **plaintext** in SQLite databases (`./db/` and `.opencode/data/`).
>
> These database files may contain sensitive information including API keys, proprietary code snippets, or personal data discussed during sessions. **Never commit these database files to a public repository.**
>
> Ensure the following patterns are listed in your `.gitignore` (they are already included by default):
> ```
> db/
> .opencode/data/
> .env
> ```

## License

MIT
