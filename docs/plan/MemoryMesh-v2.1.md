Tôi đã tiếp thu toàn bộ phản biện từ hai chuyên gia. Dưới đây là bản **MemoryMesh Plan v2.1** – tích hợp mọi điều chỉnh chi tiết, sẵn sàng cho giai đoạn 1.

---

# 🧠 MemoryMesh v2.1 – Kế hoạch Sẵn sàng Sản xuất (Production‑Ready)

**Phiên bản**: v2.1  
**Ngày:** 19/05/2026  
**Người dùng:** Shinn  
**Mục tiêu:** Xây dựng MCP server trí nhớ thông minh, local‑first, song ngữ Việt‑Anh, hoạt động bền bỉ, an toàn và hiệu năng cao ngay từ MVP.

---

## Tổng hợp thay đổi so với v2.0

| Hạng mục | Thay đổi |
|----------|----------|
| **ChromaDB** | Sửa filter syntax `where={"user_id": {"$eq": "Shinn"}}`, tạo singleton client, thêm `asyncio.Lock` ghi |
| **Embedder** | Cache model toàn cục, tải một lần duy nhất |
| **LLM metadata** | Fast‑path lưu ngay, enrichment chạy nền (background task) |
| **Recall** | Tích hợp token budget, tự giảm `top_k` hoặc cắt bớt nội dung nếu vượt ngưỡng |
| **Input Validation** | Giới hạn độ dài `text` (2000 ký tự) |
| **Response schema** | Định nghĩa JSON rõ ràng cho từng tool |
| **Graceful shutdown** | Bắt tín hiệu SIGINT/SIGTERM, đóng ChromaDB an toàn |
| **SSE transport** | Pin version MCP sau khi test, verify sớm ở GĐ1 |
| **9Router retry** | Thêm circuit breaker đơn giản (báo lỗi sau 3 lần fail liên tiếp) |

---

## 1. Mục tiêu & Nguyên lý (giữ nguyên)

- Ghi nhớ thông minh, truy xuất ngữ cảnh, tiết kiệm token.
- Học hỏi từ mâu thuẫn (GĐ4).
- Chạy hoàn toàn local, chỉ gọi LLM qua 9Router.
- Hỗ trợ tiếng Việt và tiếng Anh.

---

## 2. Kiến trúc tổng quan (cập nhật)

```
Người dùng → DeepSeek‑TUI (hoặc MCP client)
        │ (stdio / SSE)
        ▼
MemoryMesh MCP Server
   ├── MemoryManager (async lock)
   │      ├── ChromaMemoryBackend (singleton client)
   │      └── Embedder (cached model, async‑safe)
   ├── RouterClient (9Router, retry + circuit breaker)
   ├── Background Tasks (enrichment, consolidation – GĐ4)
   └── Graceful Shutdown Handler
```

---

## 3. Công nghệ & Thư viện (không đổi)

- `mcp>=1.0.0,<2.0.0` (pin version cụ thể sau test SSE)
- `chromadb` (embedded), `sentence-transformers`, `httpx`, `tiktoken`, `python-dotenv`
- Python 3.12+

---

## 4. Cấu trúc thư mục (giữ nguyên)

*(Giống v2.0, bổ sung `utils.py` nếu cần)*

---

## 5. Kế hoạch chi tiết 5 giai đoạn (điều chỉnh)

### Giai đoạn 1: Môi trường & Xác minh nền tảng (2–3 ngày)

**Mục tiêu:** Cài đặt, test mọi thành phần, phát hiện sớm các vấn đề API không tương thích.

**Các bước mới thêm vào:**

1. **Test ChromaDB filter syntax chính xác:**
   ```python
   import chromadb
   client = chromadb.PersistentClient(path="./db/chroma")
   col = client.get_or_create_collection("test")
   col.add(documents=["a"], metadatas=[{"user_id": "Shinn"}], ids=["1"])
   res = col.query(query_texts=["a"], where={"user_id": {"$eq": "Shinn"}})
   print(res)  # phải có kết quả
   ```
2. **Verify MCP SSE transport:**
   - Cài đặt `mcp` version mới nhất, chạy một server test nhỏ với `sse_server`. Dùng `curl` hoặc Python client để gọi.
   - **Pin version cứng** sau khi test thành công (vd: `mcp==1.3.0`). Ghi vào `requirements.txt`.
3. **Test mô hình embedding cache & async:**
   - Viết script nhỏ: gọi `get_embedding` 2 lần liên tiếp, đảm bảo model chỉ load một lần.
4. **Kiểm tra graceful shutdown prototype:** Thử bắt `signal.SIGINT` trong một script Python, đóng ChromaDB client.
5. Các bước cũ: test 9Router, cài thư viện, tạo `.env`, cấu trúc thư mục.

**Tiêu chí hoàn thành:** Tất cả test pass, không có lỗi API.

---

### Giai đoạn 2: Xây dựng Memory Engine & MCP Server cốt lõi (5–7 ngày)

**Mục tiêu:** MVP hoạt động ổn định với 5 tool, bảo vệ dữ liệu, xử lý lỗi tốt.

**Cập nhật chi tiết các module:**

#### 2.1 ChromaDB Backend (`chroma_backend.py`)
- **Singleton client:** Khởi tạo một lần trong `server.py`, truyền vào backend.
- **Filter đúng cú pháp:** `where={"user_id": {"$eq": user_id}}`.
- **Lock ghi:** Dùng `asyncio.Lock` trong `MemoryManager` (xem 2.5).

#### 2.2 Embedder (`embedder.py`)
- Cache model toàn cục như mẫu đã nêu:
  ```python
  _model = None
  def _get_model():
      global _model
      if _model is None:
          _model = SentenceTransformer(EMBEDDING_MODEL)
      return _model
  async def get_embedding(text):
      return await asyncio.to_thread(lambda: _get_model().encode(text).tolist())
  ```

#### 2.3 Router Client (`router_client.py`)
- Retry + fallback model như trước.
- **Circuit breaker đơn giản:** Sau 3 lần thất bại liên tiếp (kể cả retry), raise `LLMUnavailableError` thay vì tiếp tục retry. Ghi log cảnh báo. Tool handler sẽ bắt lỗi này và trả về thông báo lỗi cho người dùng.

#### 2.4 Enrichment Background Task
- Trong `MemoryManager.add_memory`:
  1. Tạo embedding, lưu ngay vào ChromaDB (fast path).
  2. Trả về ID.
  3. Dùng `asyncio.create_task(self._enrich_memory(memory_id, text))` để gọi LLM trích xuất tags, importance, summary và cập nhật metadata (không block người dùng).

#### 2.5 Memory Manager (`manager.py`)
- Thêm `self._write_lock = asyncio.Lock()`. Trong `add_memory`, `forget`, bọc các thao tác ghi ChromaDB trong `async with self._write_lock`.
- Trong `search_memory`:
  - Sau khi có kết quả, tính tổng token của các documents bằng `tiktoken`.
  - Nếu vượt ngưỡng (`MAX_RECALL_TOKENS = 1000`), giảm `top_k` hoặc cắt bớt nội dung từng document (giữ lại câu đầu).
- Input validation: kiểm tra `len(text) <= 2000`, raise ValueError nếu quá dài.

#### 2.6 MCP Tools (`tools.py`)
- Response schema rõ ràng:
  - `remember` → `{"id": "uuid", "status": "saved"}`
  - `recall` → `[{"id": "uuid", "content": "...", "score": 0.87, "tags": ["python"]}]`
  - `forget` → `{"id": "uuid", "status": "deleted"}`
  - `list_memories` → `[{"id": "uuid", "content": "...", "timestamp": "..."}]`
  - `ping` → `"pong"`

#### 2.7 Graceful Shutdown (`server.py`)
- Đăng ký handler cho `SIGINT`, `SIGTERM`:
  ```python
  import signal
  async def shutdown(signal, loop):
      logger.info("Shutting down...")
      # Đóng ChromaDB client
      chroma_client.close()  # hoặc persistant client không cần close nhưng an toàn
      loop.stop()
  loop = asyncio.get_event_loop()
  signals = (signal.SIGINT, signal.SIGTERM)
  for s in signals:
      loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))
  ```
  Hoặc dùng `try/finally` với `server.run()` nếu SDK hỗ trợ.

#### 2.8 Testing
- Test từng tool với input hợp lệ và không hợp lệ.
- Mô phỏng concurrent writes (gửi nhiều `remember` cùng lúc) để kiểm tra lock.

**Tiêu chí hoàn thành:** Server hoạt động ổn định qua stdio, xử lý lỗi mượt, không treo, không corrupt DB.

---

### Giai đoạn 3: Tích hợp DeepSeek‑TUI (1–2 ngày)

- Vẫn như cũ, nhưng ưu tiên test SSE nếu TUI chưa hỗ trợ MCP. Sử dụng version MCP đã pin.
- Viết hướng dẫn kết nối với cả 2 transport.

---

### Giai đoạn 4: Nâng cấp Premium (sau MVP)

- Các tính năng dự kiến giữ nguyên, bổ sung **Circuit breaker nâng cao** cho 9Router nếu cần.

---

### Giai đoạn 5: Kiểm thử toàn diện & Đóng gói (3–4 ngày)

- Kiểm thử tự động bao gồm các kịch bản lỗi (9Router down, DB lock, input dài).
- Benchmark token.
- Docker.

---

## 6. Quyết định kỹ thuật (bổ sung)

- **Concurrency:** `asyncio.Lock` cho mọi ghi ChromaDB; background job chỉ chạy trong cùng process.
- **Token budget:** Áp dụng từ GĐ2 trong `recall` để bảo vệ context window.
- **Circuit breaker:** Giới hạn retry, tránh treo server khi 9Router sập.
- **Validation:** Giới hạn độ dài input, tránh tràn token embedding và lãng phí lưu trữ.

---

## 7. Tiến độ (giữ nguyên)

| Giai đoạn | Thời gian |
|-----------|-----------|
| GĐ1 | 2–3 ngày |
| GĐ2 | 5–7 ngày |
| GĐ3 | 1–2 ngày |
| **Tổng MVP** | **8–12 ngày** |

---

Bản kế hoạch này đã chín muồi. Tất cả phản biện đều được xử lý triệt để. Bạn có thể bắt đầu **Giai đoạn 1** ngay bây giờ. Nếu cần tôi hỗ trợ viết code cho từng bước cụ thể, hãy nói với tôi.