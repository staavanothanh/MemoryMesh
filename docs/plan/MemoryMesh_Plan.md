# 🧠 MemoryMesh – Hệ thống Trí nhớ Thông minh cho DeepSeek

**PHiên bản**: v2.0  
**Ngày:** 19/05/2026  
**Người dùng:** Shinn  
**Mục tiêu:** Xây dựng một MCP server trí nhớ dài hạn, local‑first, song ngữ (Việt/Anh), tự động ghi nhớ và truy xuất ngữ cảnh, tiết kiệm token và học hỏi từ tương tác.

---

## 1. Mục tiêu dự án

- **Ghi nhớ thông minh:** Tự động lưu giữ các sự kiện, sở thích, kiến thức từ hội thoại.
- **Truy xuất ngữ cảnh:** Khi người dùng hỏi, trả về những ký ức liên quan nhất để LLM có ngữ cảnh đầy đủ mà không cần gửi toàn bộ lịch sử chat.
- **Tiết kiệm token:** Giảm đáng kể lượng token tiêu thụ so với việc nạp toàn bộ lịch sử.
- **Học liên tục:** Phát hiện mâu thuẫn, tự tạo quy tắc (instincts) để cải thiện chất lượng ghi nhớ.
- **Hoạt động local:** Chạy hoàn toàn trên máy cá nhân, không phụ thuộc vào dịch vụ đám mây (ngoại trừ LLM qua 9Router, nhưng 9Router cũng chạy local đến DeepSeek API).
- **Hỗ trợ song ngữ:** Xử lý tốt tiếng Việt và tiếng Anh trong cả lưu trữ và truy xuất.

---

## 2. Tổng quan kiến trúc

```
Người dùng
   |
   v
DeepSeek‑TUI (hoặc MCP client khác)
   |  (MCP protocol qua stdio hoặc SSE)
   v
MemoryMesh MCP Server
   ├── MemoryManager (abstract backend)
   │      └── ChromaDB (vector + metadata)  // GĐ2; sau đó thay thế bằng SQLite+FTS5+sqlite‑vec ở GĐ4
   ├── Embedder (local, async‑safe)
   ├── RouterClient (gọi LLM qua 9Router, có retry & fallback)
   └── Background Tasks (hooks, consolidation – GĐ4)
             |
             v
         9Router (http://127.0.0.1:20128/v1)
             |
             v
         DeepSeek API (V4 Flash / Pro)
```

**Nguyên lý hoạt động:**
- DeepSeek‑TUI kết nối đến MemoryMesh như một MCP server.
- Khi người dùng yêu cầu `remember`, MemoryMesh tính embedding của văn bản, tùy chọn gọi LLM (qua 9Router) để trích xuất metadata (tags, tóm tắt), rồi lưu vào ChromaDB.
- Khi người dùng `recall`, MemoryMesh tính embedding của truy vấn, tìm trong ChromaDB các ký ức gần nhất (theo cosine similarity), và trả về dạng văn bản.
- Mọi lời gọi LLM ra bên ngoài đều thông qua RouterClient, tích hợp retry và fallback model.
- Ở giai đoạn sau, engine sẽ được thay thế bằng hybrid search (BM25 + vector) và có cơ chế tự động hợp nhất ký ức (consolidation) và học instinct.

---

## 3. Công nghệ & Thư viện chính

| Thành phần | Công nghệ | Ghi chú |
|-----------|-----------|--------|
| **MCP Server** | `mcp` Python SDK (>=1.0.0,<2.0.0) | Hỗ trợ stdio và SSE |
| **Vector Store** | ChromaDB (embedded mode) | Nhẹ, không cần server, lưu local |
| **Embeddings** | `sentence-transformers` + `paraphrase-multilingual-MiniLM-L12-v2` | 384‑dim, hỗ trợ Việt & Anh |
| **LLM Gateway** | 9Router (OpenAI‑compatible endpoint) | Tự xây dựng, có fallback model |
| **HTTP Client** | `httpx` (async) | Gọi API tới 9Router |
| **Token Counting** | `tiktoken` | Đếm token cho DeepSeek |
| **Logging** | Python `logging` module | Ghi ra stderr + file |
| **Config** | `python‑dotenv` | Đọc biến môi trường từ file `.env` |
| **Ngôn ngữ** | Python 3.12+ | Toàn bộ project |

---

## 4. Cấu trúc thư mục

```
D:\Learning_Programing\test-project\memory-mesh\
├── .env.example                  # Mẫu biến môi trường
├── .gitignore
├── requirements.txt
├── Dockerfile                    # Giai đoạn 5
├── server.py                     # Entry point (stdio/sse)
├── config.py                     # Load config từ .env
├── logger.py                     # Cấu hình logging
├── router_client.py              # Gọi 9Router (retry, fallback)
├── embedder.py                   # Async‑safe embedding
├── prompts.py                    # System prompt cho LLM (extract metadata)
├── memory/
│   ├── __init__.py
│   ├── backend.py                # Abstract MemoryBackend
│   ├── chroma_backend.py         # ChromaDB implementation
│   └── manager.py                # Business logic (add, search, forget, list)
├── tools.py                      # MCP tool definitions & handlers
├── hooks.py                      # PostToolUse, SessionEnd hooks (GĐ4)
├── instinct_engine.py            # Học instinct (GĐ4)
├── consolidation.py              # Memory consolidation job (GĐ4)
├── tests/
│   └── test_basic.py
├── docs/
│   ├── plan.md                   # Kế hoạch này
│   ├── references/               # Phân tích các repo tham khảo
│   │   ├── mem0.md
│   │   ├── sage-memory.md
│   │   ├── LightAgent.md
│   │   ├── ReMe.md
│   │   ├── ECC.md
│   │   └── ECC_deep_dive.md
│   └── api.md                    # Mô tả tool API
└── db/                           # Thư mục lưu ChromaDB & SQLite (nếu có)
    └── chroma/
```

---

## 5. Kế hoạch chi tiết 5 giai đoạn

### Giai đoạn 1: Môi trường & Xác minh nền tảng (2–3 ngày)

**Mục tiêu:** Chuẩn bị môi trường phát triển, cài đặt thư viện, kiểm tra từng thành phần hoạt động độc lập.

**Các bước cụ thể:**

1. **Tạo virtualenv và cài đặt thư viện:**
   ```powershell
   cd D:\Learning_Programing\test-project
   python -m venv memory-mesh-env
   .\memory-mesh-env\Scripts\Activate.ps1
   pip install mcp>=1.0.0,<2.0.0 httpx chromadb sentence-transformers tiktoken python-dotenv
   ```
   (Không cài mem0ai, sqlite-vec.)

2. **Tạo file `.env`:**
   ```env
   NINEROUTER_URL=http://127.0.0.1:20128/v1
   DEFAULT_MODEL=deepseek-v4-flash
   FALLBACK_MODEL=deepseek-v4-pro
   EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
   CHROMA_DB_PATH=./db/chroma
   LOG_LEVEL=INFO
   DEFAULT_USER_ID=Shinn
   MCP_TRANSPORT=stdio
   MCP_PORT=8090
   ```

3. **Thiết lập logging (`logger.py`):**
   - Sử dụng `logging.basicConfig` với StreamHandler(stderr) và FileHandler("memorymesh.log").
   - Không dùng `print()` trong bất kỳ module nào.

4. **Test các thành phần:**
   - **9Router:** Gửi request `curl` hoặc script Python nhỏ để kiểm tra chat/completions.
   - **Embedder:** Load model `paraphrase-multilingual-MiniLM-L12-v2`, thực hiện encode một câu tiếng Việt, kiểm tra shape (1, 384).
   - **ChromaDB:** Tạo client, tạo collection, thêm một document với embedding giả, query thử.

5. **Tạo cấu trúc thư mục** như trên (các file trống, chỉ có __init__.py nếu cần).

**Tiêu chí hoàn thành:** Tất cả test đều pass, môi trường sẵn sàng cho GĐ2.

---

### Giai đoạn 2: Xây dựng Memory Engine & MCP Server cốt lõi (5–7 ngày)

**Mục tiêu:** Có một MCP server hoạt động với 5 tool (`remember`, `recall`, `forget`, `list_memories`, `ping`), sử dụng ChromaDB backend, embedding async, gọi LLM qua 9Router an toàn.

**Các module cần viết:**

#### 2.1 Abstract Backend (`memory/backend.py`)
- Định nghĩa `MemoryBackend` protocol với các phương thức: `add`, `search`, `delete`, `list_all`.
- Các tham số: `user_id` (str), `content` (str), `metadata` (dict), `embedding` (list[float]), `top_k` (int), `filter` (dict).

#### 2.2 ChromaDB Backend (`memory/chroma_backend.py`)
- Implement `MemoryBackend`.
- Khởi tạo ChromaDB client với `PersistentClient(path=CHROMA_DB_PATH)`.
- Tạo collection `memories` nếu chưa có. Cấu trúc metadata: `user_id`, `tags` (list), `importance` (int), `timestamp` (ISO 8601), `content` (text).
- `add`: nhận embedding đã tính, tạo document với ID tự động.
- `search`: trả về danh sách kết quả kèm metadata và `distance`.
- `delete`: xóa theo ID.
- `list_all`: trả về tất cả documents (có phân trang đơn giản).

#### 2.3 Embedder (`embedder.py`)
- Load model từ `EMBEDDING_MODEL` (sử dụng cache).
- Hàm `async get_embedding(text: str) -> List[float]` dùng `asyncio.to_thread`.

#### 2.4 Router Client (`router_client.py`)
- `async call_llm(prompt: str, model: str = None, retries: int = 3) -> str`
  - Dùng `httpx.AsyncClient`, timeout 30s.
  - Nếu không chỉ định model, dùng `DEFAULT_MODEL` từ config.
  - Retry với exponential backoff (2^attempt giây) khi gặp Timeout, ConnectionError, hoặc HTTP status 5xx.
  - Nếu sau tất cả retry vẫn lỗi, nếu model hiện tại là Flash thì fallback sang `FALLBACK_MODEL` và thử lại 1 lần.
  - Kiểm tra response JSON; nếu không parse được, raise `LLMError`.
- Có thể thêm hàm `extract_metadata(text: str)` gọi LLM với prompt từ `prompts.py` để lấy tags, importance, summary (tùy chọn).

#### 2.5 Memory Manager (`memory/manager.py`)
- Nhận backend (MemoryBackend) và embedder (hàm async).
- `async add_memory(text: str, tags: list[str] = None, importance: int = 3, user_id: str = DEFAULT_USER_ID) -> str`:
  1. Tạo embedding async.
  2. (Optional) Gọi `router_client.extract_metadata(text)` nếu muốn bổ sung tags/importance tự động.
  3. Chuẩn bị metadata: `user_id`, `tags`, `importance`, `timestamp`, `content` (text gốc).
  4. Gọi `backend.add`.
  5. Trả về ID của document.
- `async search_memory(query: str, top_k: int = 5, user_id: str = DEFAULT_USER_ID) -> list[dict]`:
  1. Tạo embedding của query.
  2. Gọi `backend.search(query_embedding, top_k, filter={"user_id": user_id})`.
  3. Trả về danh sách kết quả.
- `async forget_memory(memory_id: str) -> bool`: gọi backend.delete.
- `async list_memories(limit: int = 100, user_id: str = DEFAULT_USER_ID) -> list[dict]`: gọi backend.list_all với filter user_id.

#### 2.6 MCP Server & Tools (`server.py`, `tools.py`)
- Trong `server.py`:
  - Khởi tạo `Server("memorymesh")`.
  - Tạo instance của backend (ChromaMemoryBackend), embedder, router client, manager.
  - Đăng ký handlers từ `tools.py`.
  - Chọn transport dựa vào `MCP_TRANSPORT` (stdio hoặc sse). Nếu sse, dùng `sse_server`.
- Trong `tools.py`:
  - Định nghĩa các Tool object cho 5 công cụ.
  - Viết hàm handler cho từng tool, gọi manager tương ứng.
  - Xử lý lỗi, trả về `TextContent`.

**Tiêu chí hoàn thành:**
- Server khởi động không lỗi, kết nối qua stdio.
- Gọi `remember` với một câu, kiểm tra ChromaDB có dữ liệu.
- Gọi `recall` trả về kết quả liên quan.
- Gọi `forget` xóa đúng ID.
- `list_memories` liệt kê tất cả.
- `ping` trả về "pong".
- Log ghi đầy đủ, không có `print`.

---

### Giai đoạn 3: Tích hợp DeepSeek‑TUI (1–2 ngày)

**Mục tiêu:** Kết nối MemoryMesh với giao diện người dùng.

**Các bước:**
1. **Xác định khả năng MCP của DeepSeek‑TUI:** Nếu có hỗ trợ MCP, tạo file `.deepseek/mcp_servers.json` trỏ đến `server.py`. Nếu không, sử dụng transport SSE.
2. **Cấu hình SSE transport:** Chạy server với `--transport sse --port 8090`. DeepSeek‑TUI (hoặc client) gọi tool thông qua HTTP POST tới `http://localhost:8090/mcp`.
3. **Viết hướng dẫn** ngắn gọn trong `docs/`.
4. **Test end‑to‑end:** Chat một đoạn hội thoại tiếng Việt, dùng `remember`, rồi `recall` để kiểm tra.

**Tiêu chí hoàn thành:** Có thể trò chuyện với AI, MemoryMesh tự động ghi nhớ và truy xuất thông tin.

---

### Giai đoạn 4: Nâng cấp Premium (sẽ được đặc tả chi tiết sau khi MVP ổn định)

Dự kiến bao gồm:

- **Hybrid Search Engine:** Sử dụng SQLite + FTS5 (full‑text) và sqlite‑vec (vector) để thay thế ChromaDB. Kết hợp điểm BM25 và cosine similarity bằng Reciprocal Rank Fusion.
- **Memory Consolidation:** Background job chạy định kỳ, tóm tắt các memory rời rạc thành những “fact” tổng quát hơn, giảm nhiễu và tiết kiệm không gian.
- **Instinct Engine:** Học các quy tắc từ mâu thuẫn hoặc chỉnh sửa của người dùng. Lưu rule kèm confidence score, tự động áp dụng khi truy xuất.
- **Token Budget & Compaction:** Trước mỗi lần gọi LLM, ước lượng token; nếu vượt ngưỡng, tự động tóm tắt bớt ký ức ít liên quan.
- **Multi‑level Memory:** Phân biệt User Memory, Session Memory, Knowledge.
- **Xuất memory ra Markdown** (theo phong cách ReMe).

---

### Giai đoạn 5: Kiểm thử, Đóng gói & Tài liệu (3–4 ngày)

- **Kiểm thử tự động:** Viết script mô phỏng các phiên hội thoại, đánh giá độ chính xác truy xuất (precision/recall).
- **Đo lường token:** So sánh lượng token dùng khi có MemoryMesh và khi gửi full chat history.
- **Docker:** Viết Dockerfile nhẹ (python:3.12‑slim), cài đủ thư viện, copy code. Volume mount thư mục db và log.
- **Tài liệu:** README đầy đủ, hướng dẫn cài đặt, cấu hình, các tool, ví dụ sử dụng.

---

## 6. Các quyết định kỹ thuật quan trọng

- **Tại sao không dùng Mem0?** Rủi ro về độ ổn định của SQLite vector store trên Windows và việc phụ thuộc vào API nội bộ của Mem0. Tự xây dựng với ChromaDB (embedded) cho phép kiểm soát hoàn toàn, dễ dàng thay thế bằng engine khác sau này.
- **Tại sao ChromaDB mà không phải Qdrant/Weaviate?** ChromaDB có chế độ embedded, không cần process riêng, phù hợp local‑first. Về sau có thể swap sang SQLite+vec để tối giản hơn nữa.
- **Tại sao embedding local?** 9Router không hỗ trợ endpoint `/v1/embeddings`, và dùng local đảm bảo hoạt động offline hoàn toàn (trừ phần LLM). Model đa ngữ `MiniLM` đã được kiểm chứng chất lượng tốt trên cả tiếng Việt và Anh.
- **Async safety:** Mọi thao tác nặng về CPU (embedding) đều được bọc trong `asyncio.to_thread` để không block event loop của MCP.
- **Transport linh hoạt:** Hỗ trợ stdio (cho Claude Code, Cursor) và SSE (cho các client HTTP). Cấu hình qua biến môi trường.
- **user_id mặc định:** `Shinn` – đại diện cho người dùng cá nhân. Trong tương lai, có thể mở rộng lấy user_id từ JWT token hoặc request context.
- **Abstract Backend:** Cho phép thay đổi storage mà không ảnh hưởng logic bên trên. Các module khác chỉ phụ thuộc vào interface.

---

## 7. Tiến độ dự kiến

| Giai đoạn | Công việc chính | Thời gian |
|-----------|----------------|-----------|
| **GĐ1** | Cài đặt môi trường, test thành phần, tạo cấu trúc dự án | 2–3 ngày |
| **GĐ2** | Xây dựng backend, embedder, router client, memory manager, MCP server & tools | 5–7 ngày |
| **GĐ3** | Tích hợp với DeepSeek‑TUI (hoặc client), test end‑to‑end | 1–2 ngày |
| **GĐ4** | Nâng cấp premium (hybrid search, consolidation, instincts…) | Sẽ lên kế hoạch sau |
| **GĐ5** | Kiểm thử, benchmark token, Docker, tài liệu | 3–4 ngày |
| **Tổng MVP** | (GĐ1–GĐ3) | **8–12 ngày** |

---

## 8. Phụ lục: Danh sách các repo tham khảo

Các file phân tích chi tiết được lưu trong `docs/references/`. Dưới đây là tóm tắt:

- **mem0:** Multi‑level memory, entity linking, multi‑signal retrieval.  
- **sage‑memory:** Retrieval pipeline 6 giai đoạn, self‑learning loop, SQLite+vec.  
- **LightAgent:** Mẫu tích hợp mem0 vào MCP, cách define tool.  
- **ReMe:** File‑based memory, compaction 99.5%, hybrid search BM25+vector.  
- **ECC (everything‑claude‑code):** Hooks tự động, instincts engine, token budget, MCP discipline.  
- **agenticSeek:** (Tham khảo xa) Multi‑agent, local‑first.  
- **Local‑LLM‑DeepSeek‑Memory:** Fallback embedding bằng SHA‑256.

---