# mem0 — Memory Layer for AI Agents

**Repository:** [https://github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐⭐⭐⭐ (Cực kỳ cao)

## Tổng quan

Mem0 là một memory layer chuyên dụng, được thiết kế để cung cấp trí nhớ thông minh cho các AI Agents. Nó giải quyết chính xác bài toán mà MemoryMesh hướng tới: ghi nhớ, truy xuất ngữ cảnh, và tiết kiệm token qua các phiên hội thoại dài.

## Tính năng nổi bật

### 1. Multi-Level Memory (Phân cấp trí nhớ)
- **User Memory:** Lưu thông tin, sở thích người dùng xuyên suốt nhiều phiên.
- **Session Memory:** Ngữ cảnh tạm thời trong một cuộc hội thoại.
- **Agent State:** Trạng thái suy luận của agent (ví dụ: bước hiện tại trong một tác vụ phức tạp).
→ *Áp dụng cho MemoryMesh:* Thiết kế phân tầng tương tự để phân biệt dữ liệu dài hạn (user profile) và ngắn hạn (session context).

### 2. Thuật toán Memory v2 (2026)
- **Single-pass ADD-only extraction:** Trích xuất thông tin chỉ trong một lượt, không cần gọi LLM nhiều lần, tiết kiệm token.
- **Entity Linking:** Liên kết các thực thể (người, địa điểm, khái niệm) tự động, giúp truy xuất chính xác hơn.
- **Multi-signal retrieval:** Kết hợp semantic search, BM25, và entity matching để xếp hạng kết quả.
- **Temporal Reasoning:** Có khả năng suy luận về thời gian (ví dụ: "cuộc họp tuần trước", "dự án hồi tháng 3").
→ *Áp dụng:* Pipeline 3 bước của MemoryMesh có thể áp dụng trực tiếp entity linking và multi-signal retrieval để tăng độ chính xác.

### 3. API đơn giản, trực quan
- `mem0.add(text, user_id=...)` — Thêm ký ức.
- `mem0.search(query, user_id=...)` — Tìm kiếm ký ức liên quan.
- Hỗ trợ OpenAI-compatible API endpoint.
→ *Áp dụng:* Thiết kế MCP tools `remember` và `recall` theo đúng interface này để đảm bảo tính quen thuộc.

### 4. Tương thích OpenAI-compatible API
Có thể kết nối trực tiếp với bất kỳ provider nào tuân thủ định dạng OpenAI, bao gồm DeepSeek, Ollama, và **9Router** (endpoint `http://127.0.0.1:20128/v1`).
→ *Áp dụng:* MemoryMesh sẽ sử dụng 9Router làm LLM gateway cho mọi request ra ngoài.

### 5. Token Efficiency Benchmarks
- **LoCoMo:** 91.6 điểm
- **LongMemEval:** 94.8 điểm
→ *Áp dụng:* Đây là con số tham chiếu để đánh giá hiệu năng của MemoryMesh sau khi hoàn thiện.

## Hướng áp dụng vào MemoryMesh

- **Sử dụng mem0 làm backend cho Memory Engine:** Thay vì tự viết toàn bộ `memory_manager.py`, tích hợp mem0 như một thư viện để xử lý lưu trữ và truy xuất. Cách này tiết kiệm 60-70% thời gian phát triển phần lõi.
- **Tham khảo kiến trúc Multi-signal retrieval:** Kết hợp vector DB (sqlite-vec) + FTS5 + entity graph để truy xuất.
- **Áp dụng Entity Linking:** Thêm bước trích xuất thực thể khi `add_memory`, lưu vào metadata để tăng độ chính xác khi `recall`.
- **Tối ưu hóa LLM call:** Dùng cơ chế ADD-only extraction để giảm số lần gọi LLM trong pipeline bất đồng bộ.