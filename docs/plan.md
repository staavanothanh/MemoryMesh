# 🧠 MemoryMesh – Hệ thống Trí nhớ Thông minh cho LLM (và hơn thế nữa)

**Phiên bản:** v3.0 – OpenCode Edition  
**Ngày:** 19/05/2026  
**Tác giả:** Shinn  
**Mục tiêu:** Xây dựng một MCP server trí nhớ dài hạn, local‑first, song ngữ (Việt/Anh), tự động ghi nhớ và truy xuất ngữ cảnh, tiết kiệm token và học hỏi từ tương tác. Tích hợp liền mạch với OpenCode CLI để có trải nghiệm TUI đẹp, hỗ trợ đa model, tương lai sẵn sàng mở rộng.

---

## Lịch sử phiên bản

| Phiên bản | Thay đổi chính |
|-----------|----------------|
| v2.2 | Tích hợp cấu trúc thư mục chuẩn, config dataclass, custom exceptions, TypedDict schemas, structured logging. |
| v3.0 | Thay thế DeepSeek‑TUI bằng OpenCode CLI cho Giai đoạn 3. Cập nhật kiến trúc, công nghệ, kế hoạch tích hợp. |

---

## Mục lục

1. [Mục tiêu dự án](#1-mục-tiêu-dự-án)
2. [Tổng quan kiến trúc](#2-tổng-quan-kiến-trúc)
3. [Công nghệ & Thư viện chính](#3-công-nghệ--thư-viện-chính)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Kế hoạch chi tiết 5 giai đoạn](#5-kế-hoạch-chi-tiết-5-giai-đoạn)
   - [Giai đoạn 1: Môi trường & Xác minh nền tảng](#giai-đoạn-1-môi-trường--xác-minh-nền-tảng)
   - [Giai đoạn 2: Xây dựng Memory Engine & MCP Server cốt lõi](#giai-đoạn-2-xây-dựng-memory-engine--mcp-server-cốt-lõi)
   - [Giai đoạn 3: Tích hợp OpenCode CLI & End‑to‑End](#giai-đoạn-3-tích-hợp-opencode-cli--endtoend)
   - [Giai đoạn 4: Nâng cấp Premium](#giai-đoạn-4-nâng-cấp-premium)
   - [Giai đoạn 5: Kiểm thử, Đóng gói & Tài liệu](#giai-đoạn-5-kiểm-thử-đóng-gói--tài-liệu)
6. [Các quyết định kỹ thuật quan trọng](#6-các-quyết-định-kỹ-thuật-quan-trọng)
7. [Tiến độ dự kiến](#7-tiến-độ-dự-kiến)
8. [Phụ lục: Danh sách repo tham khảo](#8-phụ-lục-danh-sách-repo-tham-khảo)

---

## 1. Mục tiêu dự án

- **Ghi nhớ thông minh:** Tự động lưu giữ các sự kiện, sở thích, kiến thức từ hội thoại.
- **Truy xuất ngữ cảnh:** Khi người dùng hỏi, trả về những ký ức liên quan nhất để LLM có ngữ cảnh đầy đủ mà không cần gửi toàn bộ lịch sử chat.
- **Tiết kiệm token:** Giảm đáng kể lượng token tiêu thụ so với việc nạp toàn bộ lịch sử.
- **Học liên tục:** Phát hiện mâu thuẫn, tự tạo quy tắc (instincts) để cải thiện chất lượng ghi nhớ.
- **Hoạt động local:** Chạy hoàn toàn trên máy cá nhân, không phụ thuộc vào dịch vụ đám mây (ngoại trừ LLM qua 9Router, nhưng 9Router cũng chạy local đến DeepSeek API).
- **Hỗ trợ song ngữ:** Xử lý tốt tiếng Việt và tiếng Anh trong cả lưu trữ và truy xuất.
- **Giao diện TUI hiện đại:** Sử dụng OpenCode CLI – đẹp, hỗ trợ đa model, chuyển đổi model linh hoạt, MCP native.

---

## 2. Tổng quan kiến trúc

```
Người dùng
   |
   v
OpenCode CLI (hoặc MCP client khác)
   |  (MCP protocol qua stdio)
   v
MemoryMesh MCP Server
   ├── MemoryManager (async lock)
   │      ├── ChromaMemoryBackend (singleton client, audit logs)
   │      └── Embedder (cached model, async‑safe)
   ├── RouterClient (9Router, retry + circuit breaker)
   ├── Background Tasks (enrichment, consolidation – GĐ4)
   └── Graceful Shutdown Handler
                |
                v
            9Router (http://127.0.0.1:20128/v1)
                |
                v
      LLM Provider (DeepSeek, OpenAI, Anthropic, Google, Ollama...)
```

**Nguyên lý hoạt động:**
- OpenCode kết nối đến MemoryMesh như một MCP server, khai báo trong file config.
- Khi người dùng gọi `@memorymesh remember`, MemoryMesh tính embedding, lưu ngay (fast path), sau đó chạy background task gọi LLM để trích xuất metadata.
- Khi người dùng `@memorymesh recall`, MemoryMesh tính embedding, tìm ký ức gần nhất, áp dụng token budget, và trả về kết quả để OpenCode (và LLM đang sử dụng) có thể dùng làm ngữ cảnh.
- OpenCode cho phép chuyển đổi model nhanh chóng (ví dụ: GPT‑4o cho phân tích, DeepSeek‑V4 cho code), nhưng MemoryMesh vẫn hoạt động trong suốt với mọi model.

---

## 3. Công nghệ & Thư viện chính

| Thành phần | Công nghệ | Ghi chú |
|-----------|-----------|--------|
| **TUI Client** | **OpenCode CLI** (v3.0) | Đẹp, đa model, MCP native, 27K+ stars |
| **MCP Server** | `mcp` Python SDK (>=1.0.0,<2.0.0) | Hỗ trợ stdio (cho OpenCode) và SSE |
| **Vector Store** | ChromaDB (embedded mode) | Nhẹ, không cần server, lưu local |
| **Embeddings** | `sentence-transformers` + `paraphrase-multilingual-MiniLM-L12-v2` | 384‑dim, hỗ trợ Việt & Anh |
| **LLM Gateway** | 9Router (OpenAI‑compatible endpoint) | Tự xây dựng, có fallback model |
| **HTTP Client** | `httpx` (async) | Gọi API tới 9Router |
| **Token Counting** | `tiktoken` | Đếm token cho DeepSeek (và các model khác nếu cần) |
| **Logging** | Python `logging` + `RotatingFileHandler` | Ghi ra stderr + file xoay vòng |
| **Config** | `python‑dotenv` + dataclass | Đọc biến môi trường, có validation |
| **Testing** | `pytest`, `pytest-asyncio` | Unit test + integration test |
| **Ngôn ngữ** | Python 3.12+ | Toàn bộ project |

---

## 4. Cấu trúc thư mục

```
D:\Learning_Programing\MemoryMesh\memory-mesh
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── src/
│   └── memorymesh/
│       ├── __init__.py
│       ├── main.py                  # Entry point (stdio/sse)
│       ├── config.py                # AppConfig dataclass
│       ├── errors.py                # Custom exceptions
│       ├── types.py                 # TypedDict schemas
│       ├── logging_.py              # Logging setup
│       ├── embedder.py              # Singleton embedder
│       ├── router.py                # RouterClient (9Router)
│       ├── prompts.py               # System prompts
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── backend.py           # Protocol
│       │   ├── chroma_impl.py       # ChromaDB implementation
│       │   └── manager.py           # Business logic
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py            # Server init, transport
│       │   ├── tools.py             # Tool definitions
│       │   └── handlers.py          # Handler functions
│       └── hooks.py                 # PostToolUse (stub for GĐ4)
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_embedder.py
│   │   ├── test_router.py
│   │   ├── test_memory_manager.py
│   │   └── test_chroma_backend.py
│   ├── integration/
│   │   ├── test_mcp_server.py
│   │   └── test_e2e_workflow.py
│   └── fixtures/
├── docs/
│   ├── plan.md
│   ├── api.md
│   ├── opencode_setup.md            # Hướng dẫn cấu hình OpenCode
│   └── references/
├── scripts/
│   ├── verify_env.py
│   └── benchmark.py
├── opencode/
│   └── config.example.json          # File config mẫu cho OpenCode
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 5. Kế hoạch chi tiết 5 giai đoạn

### Giai đoạn 1: Môi trường & Xác minh nền tảng (2–3 ngày)

**Mục tiêu:** Chuẩn bị môi trường phát triển, cài đặt thư viện, kiểm tra từng thành phần hoạt động độc lập, phát hiện sớm các vấn đề không tương thích.

**Các bước cụ thể:**

1. **Tạo virtualenv và cài đặt thư viện:**
   ```powershell
   cd D:\Learning_Programing\test-project
   python -m venv memory-mesh-env
   .\memory-mesh-env\Scripts\Activate.ps1
   pip install mcp>=1.0.0,<2.0.0 httpx chromadb sentence-transformers tiktoken python-dotenv pytest pytest-asyncio
   ```

2. **Tạo file `.env` (theo cấu trúc AppConfig):**
   ```env
   ROUTER_URL=http://127.0.0.1:20128/v1
   DEFAULT_MODEL=ds/deepseek-v4-flash
   FALLBACK_MODEL=ds/deepseek-v4-pro
   ROUTER_TIMEOUT=30
   ROUTER_RETRIES=3
   EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
   CHROMA_DB_PATH=./db/chroma
   LOG_LEVEL=INFO
   DEFAULT_USER_ID=Shinn
   MCP_TRANSPORT=stdio
   MCP_PORT=8090
   MAX_MEMORY_LENGTH=2000
   ```

3. **Tạo cấu trúc thư mục như trên** (các file trống, chỉ có `__init__.py`).

4. **Test các thành phần:**
   - **9Router:** Gửi request `curl` hoặc script Python nhỏ để kiểm tra chat/completions.
   - **Embedder:** Load model, kiểm tra shape (1,384) và cache.
   - **ChromaDB filter syntax:** Test query với `where={"user_id": {"$eq": "Shinn"}}`.
   - **MCP stdio transport:** Chạy server test nhỏ.
   - **Graceful shutdown prototype:** Bắt SIGINT, đóng ChromaDB client.

5. **Tạo các file nền tảng:** `config.py`, `errors.py`, `types.py`, `logging_.py` (xem mã mẫu ở cuối phần 5.1).

**Tiêu chí hoàn thành:** Tất cả test đều pass, môi trường sẵn sàng cho GĐ2.

---

### Giai đoạn 2: Xây dựng Memory Engine & MCP Server cốt lõi (5–7 ngày)

**Mục tiêu:** Có một MCP server hoạt động với 5 tool (`remember`, `recall`, `forget`, `list_memories`, `ping`), sử dụng ChromaDB backend, embedding async, gọi LLM qua 9Router an toàn, có xử lý lỗi và bảo vệ dữ liệu.

**Các module cần viết:**

- **Abstract Backend** (`memory/backend.py`): định nghĩa protocol.
- **ChromaDB Backend** (`memory/chroma_impl.py`): implement, singleton client, filter đúng cú pháp, thêm collection `audit_logs`.
- **Embedder** (`embedder.py`): cache model toàn cục, `get_embedding` async.
- **Router Client** (`router.py`): retry + fallback + circuit breaker, `extract_metadata`.
- **Memory Manager** (`memory/manager.py`): lock ghi, fast path + background enrichment, token budget trong recall, input validation.
- **MCP Tools & Handlers** (`mcp/tools.py`, `mcp/handlers.py`, `mcp/server.py`): định nghĩa tool, xử lý lỗi, response schema chuẩn, graceful shutdown.

**Tiêu chí hoàn thành:** Server hoạt động ổn định qua stdio, các tool trả về đúng schema, log đầy đủ, unit test pass.

---

### Giai đoạn 3: Tích hợp OpenCode CLI & End‑to‑End (1–2 ngày)

**Mục tiêu:** Kết nối MemoryMesh MCP server với OpenCode CLI, cho phép người dùng tương tác qua giao diện terminal đẹp, sử dụng đồng thời nhiều model và gọi các công cụ nhớ một cách trong suốt.

**Các bước thực hiện:**

1. **Cài đặt OpenCode CLI:**
   ```powershell
   # Windows (dùng Scoop hoặc tải binary)
   scoop install opencode
   # Hoặc tải từ GitHub Releases: https://github.com/opencode-ai/opencode/releases
   ```

2. **Tạo file cấu hình OpenCode (`~/.opencode/config.json`):**
   ```json
   {
     "providers": {
       "deepseek": {
         "provider": "deepseek",
         "api_key": "your-api-key",
         "base_url": "http://127.0.0.1:20128/v1",
         "models": ["ds/deepseek-v4-flash", "ds/deepseek-v4-pro"],
         "default_model": "ds/deepseek-v4-flash"
       },
       "openai": {
         "provider": "openai",
         "api_key": "your-openai-key",
         "models": ["gpt-4o", "gpt-4-turbo"]
       },
       "anthropic": {
         "provider": "anthropic",
         "api_key": "your-anthropic-key",
         "models": ["claude-opus-4.6"]
       }
     },
     "mcp_servers": {
       "memorymesh": {
         "command": "python",
         "args": [
           "D:\\Learning_Programing\\test-project\\memory-mesh\\src\\memorymesh\\main.py"
         ],
         "env": {
           "ROUTER_URL": "http://127.0.0.1:20128/v1",
           "DEFAULT_MODEL": "ds/deepseek-v4-flash",
           "FALLBACK_MODEL": "ds/deepseek-v4-pro",
           "EMBEDDING_MODEL": "paraphrase-multilingual-MiniLM-L12-v2",
           "CHROMA_DB_PATH": "D:\\Learning_Programing\\test-project\\memory-mesh\\db\\chroma",
           "DEFAULT_USER_ID": "Shinn",
           "MCP_TRANSPORT": "stdio"
         },
         "disabled": false
       }
     },
     "default_provider": "deepseek"
   }
   ```

3. **Khởi động OpenCode và kiểm tra kết nối:**
   ```bash
   opencode
   # Trong giao diện OpenCode, kiểm tra MCP server đã load:
   # /mcp list
   # Gọi thử:
   @memorymesh ping
   # Kết quả mong đợi: pong
   ```

4. **Kiểm thử End‑to‑End với OpenCode:**
   - Chat bằng tiếng Việt: `> Tôi tên là Khang, thích phát triển AI với Python.`
   - Yêu cầu ghi nhớ: `@memorymesh remember "Tên tôi là Khang, thích AI với Python" --tags "giới thiệu,ai"`
   - Hỏi lại với model bất kỳ: `@memorymesh recall "tên và sở thích của tôi"` → MemoryMesh trả về ký ức.
   - Chuyển model: `/model gpt-4o`, hỏi tiếp nhưng ký ức vẫn được truy xuất chính xác.

5. **Viết hướng dẫn trong `docs/opencode_setup.md`:** cài đặt, cấu hình model, thêm MCP server, workflow mẫu.

**Tiêu chí hoàn thành:** OpenCode kết nối thành công, người dùng có thể nhớ và truy xuất qua các tool, model switching hoạt động mượt mà.

---

### Giai đoạn 4: Nâng cấp Premium (sau MVP)

Dự kiến bao gồm:
- Hybrid Search Engine (SQLite + FTS5 + sqlite-vec)
- Memory Consolidation (tóm tắt, hợp nhất ký ức)
- Instinct Engine (học quy tắc từ mâu thuẫn)
- Token Budget & Compaction nâng cao
- Multi‑level Memory (User, Session, Knowledge)

---

### Giai đoạn 5: Kiểm thử, Đóng gói & Tài liệu (3–4 ngày)

- Mở rộng bộ test với các kịch bản lỗi, kiểm thử trên OpenCode.
- Đo lường token, so sánh hiệu quả.
- Dockerfile và docker-compose.yml.
- Tài liệu README, hướng dẫn cài đặt, API, opencode_setup.md.

---

## 6. Các quyết định kỹ thuật quan trọng

- **Chọn OpenCode thay vì DeepSeek‑TUI:** Hỗ trợ đa model (75+ provider), MCP native, TUI đẹp, cộng đồng lớn, không bị khóa vào một nhà cung cấp LLM.
- **Không dùng Mem0:** Rủi ro ổn định, tự xây dựng để kiểm soát hoàn toàn.
- **ChromaDB:** Embedded, nhẹ, dễ thay thế sau.
- **Embedding local:** 9Router không có endpoint embeddings, đảm bảo offline.
- **Async safety, lock, graceful shutdown, token budget, circuit breaker:** Giúp hệ thống bền bỉ ngay từ MVP.
- **Abstract backend:** Dễ dàng swap storage engine sau.

---

## 7. Tiến độ dự kiến

| Giai đoạn | Công việc chính | Thời gian |
|-----------|----------------|-----------|
| GĐ1 | Môi trường, test thành phần, file nền tảng | 2–3 ngày |
| GĐ2 | Xây dựng memory engine, MCP server | 5–7 ngày |
| **GĐ3** | **Tích hợp OpenCode CLI, cấu hình, test end‑to‑end** | **1–2 ngày** |
| GĐ4 | Nâng cấp premium | Sau MVP |
| GĐ5 | Kiểm thử, Docker, tài liệu | 3–4 ngày |
| **Tổng MVP** | (GĐ1–GĐ3) | **8–12 ngày** |

---

## 8. Phụ lục: Danh sách repo tham khảo

- **mem0:** Multi‑level memory, entity linking.
- **sage‑memory:** Retrieval pipeline, self‑learning.
- **LightAgent:** MCP tool định nghĩa.
- **ReMe:** Hybrid search, compaction.
- **ECC (everything‑claude‑code):** Hooks, instincts, token budget.
- **agenticSeek, Local‑LLM‑DeepSeek‑Memory:** Tham khảo thêm.
