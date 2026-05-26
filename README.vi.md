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

**Máy chủ MCP bộ nhớ bền vững (persistent memory), ưu tiên máy cục bộ (local-first) cho các AI agent.** v0.5.0 mang đến Đồ thị tri thức (Knowledge Graph), Học tập hành vi (Behavioral Learning), Tự động hóa vòng đời (Lifecycle Automation), và Quản lý ngữ cảnh động (Dynamic Context Management).

## Quick Start

```bash
# Cài đặt bản rút gọn (dùng API embedding từ xa)
pip install memorymesh
# Hoặc với mô hình nhúng cục bộ (khuyên dùng cho chế độ offline)
pip install "memorymesh[local]"

python -m memorymesh init   # thiết lập một lệnh
opencode
```

### Công cụ CLI

```bash
memorymesh sessions --limit 20   # liệt kê các phiên gần đây
memorymesh stats                 # hiển thị thống kê hệ thống
memorymesh init                  # khởi động workspace
```

## Có gì mới trong v0.5.0

| Tính năng | Mô tả |
|---------|-------------|
| **🧠 Đồ thị tri thức** | Các thực thể/quan hệ SQLite với `trace_entity`, `query_graph`, `create_entity`, `create_relation`. Recursive CTE cho suy luận đa chặng. Đầu ra XML Triplet. An toàn cho chế độ Plan/Read-Only. |
| **📜 Lịch sử nguyên bản (Lossless)** | Ghi nhật ký công cụ nguyên văn với nén zlib qua AsyncBatchLogger. Truy xuất với `recall_raw` — không còn mất ngữ cảnh do nén. |
| **🤖 Học tập hành vi (Bản năng v2)** | RAM cache với regex tiền biên dịch cho khớp O(1). Trích xuất mẫu workflow N-gram. Tự động áp dụng tag khi độ tin cậy > 0.8. Bản năng theo phạm vi dự án. |
| **⚡ Tự động hóa vòng đời** | Context Delta nén (<800 tokens) khi tự động recall. Tóm tắt lười biếng (Orphan Recovery) cho các cột mốc bị thiếu. Idle watchdog (15 phút). |
| **📄 Quản lý ngữ cảnh động** | Phân trang Keyset (cursor). Điểm số động đẩy vào SQLite CTEs. Đếm token đa luồng. |
| **🔧 Giảm tải Embedding** | Core nhẹ: `pip install memorymesh` (không PyTorch). Tùy chọn `[local]` cho SentenceTransformer. Hoặc dùng API embedding từ xa. |
| **🛡️ Tính vững chắc** | `_safe_task_wrapper` cho mọi tác vụ nền. Retry với exponential backoff. Phát hiện lỗi im lặng & ghi log ở mức `ERROR`. |

## Kiến trúc

```mermaid
graph TD
    LLM[Agent / LLM] -->|call_tool| MM[MemoryMesh MCP Server]

    subgraph MM[MemoryMesh Server]
        VEC[(sqlite-vec ANN)] --- FTS[(FTS5 Full-Text)]
        VEC --- G[(Đồ thị tri thức entities/relations)]
        RAW[(Lịch sử nguyên bản Log)] --- SESS[(Lưu trữ Phiên)]
        INST[(RAM Cache Bản năng)] --- MIDDLEWARE[Tool Execution Middleware]
    end

    MM -->|recall(query, cursor?)| LLM
    LLM -->|create_entity / create_relation| G
```

**Thiết kế chính:**
- SQLite đơn với sqlite-vec cho tìm kiếm vector + FTS5 cho từ khóa + Bảng đồ thị cho các mối quan hệ.
- Các tác vụ nền (enrichment, consolidation, trích xuất sự kiện) được giới hạn tốc độ với tự động retry.
- Ký ức cấp phiên tự động hết hạn sau 7 ngày.
- Mọi tác vụ nền được bao bọc trong `_safe_task_wrapper` — không có lỗi im lặng.

### 22 Công cụ MCP

| Phân loại | Công cụ |
|----------|-------|
| **Bộ nhớ** | `remember`, `recall`, `forget`, `archive_memory`, `unarchive_memory`, `list_memories` |
| **Đồ thị tri thức** | `create_entity`, `create_relation`, `query_graph`, `trace_entity` |
| **Vòng đời phiên** | `new_session`, `end_session`, `resume_session`, `delete_session`, `list_sessions`, `get_session_context`, `save_system_prompt`, `preserve_session_memories` |
| **Workspace** | `commit_milestone`, `save_workspace_context` |
| **Học tập** | `learn_session`, `recall_raw` |
| **Tiện ích** | `ping` |

### Tinh hoa ẩn (Hidden Gems)

- **Ngữ cảnh tức thời (Optimistic Hydration):** Tính toán trước các mỏ neo ngữ nghĩa trước khi đóng phiên. Lưu trong RAM Cache — AI lấy lại ngữ cảnh trong `<5ms`.
- **Truy xuất 3 tầng dự phòng:** 1. Ngữ nghĩa (sqlite-vec) → 2. FTS5 từ khóa → 3. Quét thời gian. Không có "ảo giác do mất trí nhớ".
- **Cơ chế điểm nghẽn (Choke Point):** Sau 5+ action chưa commit, `recall` bị chặn cho đến khi gọi `commit_milestone`. Ngăn tràn ngữ cảnh.
- **GraphRAG qua SQLite CTE:** Duyệt quan hệ đa chặng dùng SQL `WITH RECURSIVE` — không cần DB đồ thị ngoài.
- **Tiêm Bản năng JIT:** Middleware thực thi công cụ với cửa sổ trượt (deque maxlen=5). Khớp với bản năng regex tiền biên dịch trong micro giây.
- **Phân trang hỗn hợp:** Phân trang Keyset với điểm số động đẩy vào SQLite — hiệu suất trang sâu O(1).

## Cài đặt

### Yêu cầu
- Python **3.12+**
- Một endpoint LLM tương thích OpenAI

### Cài đặt rút gọn (Remote Embedding)
```bash
pip install memorymesh
```
Core chỉ ~50MB. Embedding tính toán qua API từ xa.

### Cài đặt đầy đủ (Local Embedding)
```bash
pip install "memorymesh[local]"
```
Bao gồm `sentence-transformers` (~1.5GB). Embedding tính toán cục bộ — hoàn toàn offline.

### Với CLI (Bảng Rich)
```bash
pip install "memorymesh[cli]"
```

### Phát triển
```bash
pip install -e ".[test,local,cli]"
```

### Môi trường

Sao chép `.env.example` thành `.env` và chỉnh sửa:

| Biến | Mặc định | Mô tả |
|----------|---------|-------------|
| `ROUTER_URL` | `http://127.0.0.1:20128/v1` | LLM endpoint |
| `DEFAULT_MODEL` | `your-model` | Model LLM chính |
| `BACKGROUND_MODEL_POOL` | — | Danh sách model nền (phẩy) |
| `EMBEDDING_MODE` | `local` | `local` hoặc `remote` |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Model nhúng cục bộ |
| `VEC_DB_PATH` | `./db/memory.db` | Đường dẫn SQLite |
| `AUTO_EXTRACT_FACTS` | `true` | Tắt trích xuất sự kiện tự động |
| `DEFAULT_USER_ID` | `your_user_id` | Người dùng mặc định |

## Tham chiếu CLI

### `memorymesh sessions`
Liệt kê phiên gần đây với trạng thái, workspace, dấu thời gian.

### `memorymesh stats`
Hiển thị thống kê hệ thống: số phiên, ký ức, đồ thị, kích thước DB.

### `memorymesh init`
Thiết lập workspace một lệnh — tạo `.env`, thư mục `db/`, cấu hình MCP.

## Kiểm thử

```bash
make test          # chạy tất cả
make test-unit     # unit test
make test-int      # integration test
```

## Cấu trúc Dự án

```
src/memorymesh/
  cli.py                  CLI tools (sessions, stats, init)
  config.py               App configuration
  router.py               LLM router
  embedder.py             Embedding interface
  embeddings/             Factory & Providers
  memory/                 Core CRUD + background tasks
    sqlite_vec_backend.py Single-DB
    graph_store.py        Knowledge Graph + CTE
    context_manager.py    Keyset pagination
    instinct_manager.py   Instinct RAM cache + regex
    session_store.py      Session lifecycle + raw log
    async_batch_logger.py zlib-compressed logs
  mcp_server/             Server + handlers
tools/
    tools.py              Tool schema definitions
tests/
    281+ tests (pytest, asyncio_mode=auto)
```

## Bảo mật

> [!CAUTION]
> **QUAN TRỌNG:** MemoryMesh lưu dữ liệu dạng **văn bản thuần** trong SQLite (`./db/`). Không commit lên repo công khai.
> ```
> db/
> .env
> ```

## Bảo trì

### Xây dựng lại Chỉ mục Vector
```bash
python scripts/rebuild_vec.py
```

## Giấy phép
MIT
