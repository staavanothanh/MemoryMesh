# ReMe — Remember Me, Refine Me (AgentScope Memory Kit)

**Repository:** [https://github.com/agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐⭐⭐ (Rất cao)

## Tổng quan

ReMe là bộ công cụ quản lý trí nhớ do nhóm AgentScope phát triển. Nó cung cấp hai chế độ: file-based (ReMeLight) dùng Markdown thuần túy cho con người đọc được, và vector-based dùng embedding. Điểm đặc biệt là khả năng nén ký ức siêu hiệu quả và quản lý context window.

## Tính năng nổi bật

### 1. File-Based Memory (ReMeLight)
- Lưu trữ ký ức dưới dạng file Markdown (`MEMORY.md`, `memory/YYYY-MM-DD.md`).
- Người dùng có thể đọc, chỉnh sửa, sao chép trực tiếp bằng editor.
→ *Áp dụng:* MemoryMesh có thể thêm tính năng xuất/nhập memory dưới dạng Markdown để người dùng kiểm soát, chỉnh sửa thủ công khi cần (tăng transparency).

### 2. Context Management & Memory Compaction
- Tự động giám sát context window (tổng số token).
- Khi vượt ngưỡng, thực hiện **compact_memory**: nén 223,838 tokens → 1,105 tokens (tỷ lệ nén 99.5%).
→ *Áp dụng:* MemoryMesh sẽ có tính năng lazy loading và compact tự động tương tự, đặc biệt hữu ích khi gọi LLM qua 9Router để tiết kiệm token tối đa.

### 3. Hybrid Search (Vector + BM25)
- Kết hợp cả semantic search (vector) và keyword search (BM25) để tăng recall.
- Cho phép truy vấn bằng ngôn ngữ tự nhiên lẫn từ khóa chính xác.
→ *Áp dụng:* Khẳng định thêm cho chiến lược hybrid retrieval mà MemoryMesh sẽ áp dụng (tương tự sage-memory và mem0).

### 4. Nhật ký hội thoại có cấu trúc
- `dialog/YYYY-MM-DD.jsonl`: Lưu trữ log hội thoại thô, có thể dùng để tái tạo ngữ cảnh hoặc huấn luyện.
- `memory/YYYY-MM-DD.md`: Tóm tắt hội thoại hàng ngày.
→ *Áp dụng:* MemoryMesh sẽ tổ chức dữ liệu tương tự: `logs/` chứa dữ liệu thô, `memories/` chứa ký ức đã qua xử lý (summary, keywords, embedding).

### 5. Tính năng "Refine"
Cho phép agent tự đánh giá và điều chỉnh lại ký ức dựa trên thông tin mới, tránh mâu thuẫn.
→ *Áp dụng:* Cộng hưởng với self-learning loop của sage-memory để giải quyết bài toán mâu thuẫn thông tin.

## Hướng áp dụng vào MemoryMesh

- **Áp dụng cấu trúc file tổ chức:** Tạo thư mục `memory/` và `logs/` với quy ước đặt tên theo ngày tháng.
- **Áp dụng cơ chế compaction:** Khi tổng token của ký ức truy xuất vượt ngưỡng, tự động tóm tắt trước khi đưa vào prompt.
- **Xuất memory ra Markdown:** Cho phép người dùng xem và sửa `MEMORY.md` qua tool `export_memory`.
- **Hybrid search mặc định:** Dùng BM25 cho truy vấn ngắn, vector cho ngữ nghĩa, kết hợp điểm số.