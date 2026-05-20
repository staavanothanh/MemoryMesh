# MemoryMesh — Kế hoạch phát triển

**Version:** v3.0  
**Ngày:** 20/05/2026  
**Tác giả:** Shinn  
**Mục tiêu:** MCP server trí nhớ dài hạn, local‑first, song ngữ Việt/Anh, tự động ghi nhớ và truy xuất ngữ cảnh một cách thông minh qua cơ chế **Atomic Facts + Dynamic Turn-by-Turn Recall**.

---

## 1. Hiện trạng (✅ Đã hoàn thành)

### Giai đoạn 1 — Core Engine
| # | Module | Mô tả |
|---|--------|-------|
| ✅ | ChromaDB Backend | Vector store, filter user/level, audit logs |
| ✅ | FTS5 Backend | Full-text search, level column, schema migration |
| ✅ | Hybrid Backend | ChromaDB + FTS5 song song, weighted RRF fusion |
| ✅ | Embedder | SentenceTransformer async, cached singleton |
| ✅ | Router Client | 9Router HTTP, retry + fallback + circuit breaker |
| ✅ | Memory Manager | CRUD, enrichment, consolidation hook, token budget |
| ✅ | HookRegistry | Pub-sub event system |
| ✅ | Consolidation Engine | Cosine sim clustering, LLM merge |
| ✅ | 120 tests | Unit + integration |
| ✅ | 13 MCP tools | remember, recall, forget, list_memories, ping, save_system_prompt, save_context_pair, list_sessions, get_session_context, new_session, end_session, save_workspace_context, resume_session |

### Giai đoạn 2 — Session Lifecycle
| # | Module | Mô tả |
|---|--------|-------|
| ✅ | SessionStore | SQLite sessions + context_log + workspace_snapshots |
| ✅ | CodebaseScanner | Directory tree, key files, auto-scan on start |
| ✅ | Conversation Auto-Save | _auto_log ghi context_log + ChromaDB memory |
| ✅ | Session Lifecycle | new_session, end_session, auto-close stale |

### Giai đoạn 3 — Cross-session Recall
| # | Module | Mô tả |
|---|--------|-------|
| ✅ | On-startup Recall | _auto_recall_context (knowledge + user) |
| ✅ | Context Compaction | _compact_session — LLM summarize khi end |
| ✅ | resume_session tool | Khôi phục context session cũ |

---

## 2. Vấn đề cốt lõi & Hướng giải quyết

### Vấn đề
Cơ chế hiện tại là **State Transfer** (chuyển trạng thái), không phải **Memory Retrieval**:
- Auto-recall nạp memories vào context window session mới
- Session càng nhiều → càng đầy context
- Không khác gì nạp nguyên lịch sử cũ

### Giải pháp (thống nhất từ chuyên gia)

```
Session cũ: raw conversation (5000 tokens)
    ↓ LLM extract atomic facts ↓
Kho memory: atomic facts (150 tokens)
    ↓
Session mới: context = 0 (hoàn toàn trống)
    ↓ User gửi message
    ↓ Top model tự quyết định: cần recall hay không?
       ↓ Nếu cần → tự gọi recall tool
       ↓ Recall chỉ trả về top 3-5 facts liên quan (~200 tokens)
    ↓ Top model nhận facts + trả lời
```

### 3 trụ cột chính

#### Trụ cột A — Atomic Fact Extraction
- **Không** lưu raw conversation vào memory
- Mỗi lần save_context_pair → lưu raw **và** chạy LLM background extract atomic facts
- Atomic fact = câu khẳng định ngắn, độc lập, chứa 1 thông tin duy nhất
- Consolidation: ghi đè fact cũ nếu mâu thuẫn, xoá fact lỗi thời

#### Trụ cột B — Dynamic Turn-by-Turn Recall
- **Không** preload bất cứ gì vào session mới
- Mỗi turn user nhập message → top model tự suy nghĩ: "prompt này có cần old memories không?"
- Nếu cần → tự gọi `recall` → chỉ lấy top facts liên quan
- Kết quả recall là những atomic fact đã được sàng lọc, không phải raw conversation

#### Trụ cột C — Two-Tier Model
- **Model tầng thấp (MemoryMesh server):** Nhận query từ recall, tìm top facts liên quan từ ChromaDB, trả về
- **Model tầng cao (OpenCode's LLM):** Quyết định có cần recall không, nhận facts từ tầng thấp, sinh câu trả lời
- Không cần thêm model thứ 3 — MCP tool recall đã đóng vai trò "tầng thấp"

---

## 3. Kế hoạch chi tiết

### 🎯 Phase A — Refactor: Atomic Fact Engine

**Mục tiêu:** Thay đổi cách lưu memory từ raw conversation sang atomic facts

| # | Công việc | File | Mô tả |
|---|-----------|------|-------|
| A1 | Thêm ATOMIC_FACT_EXTRACT_PROMPT | prompts.py | Prompt LLM extract atomic facts từ conversation |
| A2 | Tạo FactExtractor (async background) | memory/fact_extractor.py | Gọi LLM extract facts, tách câu, validate uniqueness |
| A3 | Sửa save_context_pair | handlers.py | Lưu raw + trigger FactExtractor background |
| A4 | Thêm consolidate_facts | consolidation.py | Ghi đè fact mâu thuẫn, xoá fact lỗi thời |
| A5 | Sửa search_memory ưu tiên facts | manager.py | Ưu tiên trả về atomic facts hơn raw conversation |

### 🎯 Phase B — Refactor: Dynamic Recall

**Mục tiêu:** Context session mới hoàn toàn trống, chỉ recall khi cần

| # | Công việc | File | Mô tả |
|---|-----------|------|-------|
| B1 | Xoá _auto_recall_context preload | handlers.py | Không preload session memories khi start |
| B2 | Thêm recall instruction vào system prompt | prompts.py | Hướng dẫn model tự gọi recall khi cần |
| B3 | Giữ _auto_recall_context cho knowledge+user | handlers.py | Chỉ preload knowledge + user, tối đa 3 items |
| B4 | Tối ưu recall response format | handlers.py | Trả về dạng atomic fact bullet, không raw |

### 🎯 Phase C — Infrastructure

| # | Công việc | File | Mô tả |
|---|-----------|------|-------|
| C1 | Dockerfile | docker/Dockerfile | Đóng gói server |
| C2 | docker-compose.yml | docker/docker-compose.yml | Orchestration |
| C3 | SSE transport support | server.py | Thêm transport SSE bên cạnh stdio |
| C4 | Graceful shutdown | server.py | SIGTERM/SIGINT cleanup |
| C5 | pyproject.toml hoàn chỉnh | pyproject.toml | Entry points, dependencies |

### 🎯 Phase D — Instinct Engine

| # | Công việc | File | Mô tả |
|---|-----------|------|-------|
| D1 | Instinct table + CRUD | memory/instinct_store.py | SQLite lưu quy tắc tự học |
| D2 | Instinct learning loop | memory/instinct.py | Phát hiện pattern, sinh quy tắc |
| D3 | Instinct integration | manager.py | Áp dụng quy tắc khi remember/recall |

---

## 4. Luồng hoạt động mới (đích đến)

```
Session mới khởi tạo
    │
    ├── Auto-create session ID (nhẹ, 0 memory preload)
    ├── Codebase snapshot (level=knowledge, 1 memory)
    └── Sẵn sàng nhận tin nhắn
         │
         ▼
[User] "code tiếp RRF nhé"
    │
    ▼
[Top model — OpenCode LLM]
    │ suy nghĩ: "Chủ đề này có trong memory cũ không?"
    │
    ├── Nếu KHÔNG → trả lời bình thường (0 token memory)
    │
    └── Nếu CÓ → tự gọi recall(query="RRF code tiếp")
         │
         ▼
    [MemoryMesh — tầng thấp]
         │ tìm atomic facts liên quan nhất
         ▼
    Trả về: 3 atomic facts (dạng bullet, ~200 tokens)
         │
         ▼
    [Top model] nhận facts + prompt user → trả lời
```

---

## 5. Tiến độ dự kiến

| Phase | Công việc | Thời gian |
|-------|-----------|-----------|
| **A** | Atomic Fact Engine | ~2-3 sessions |
| **B** | Dynamic Recall | ~1-2 sessions |
| **C** | Infrastructure | ~2-3 sessions |
| **D** | Instinct Engine | ~2-3 sessions |
| **Tổng** | | **~7-11 sessions** |
