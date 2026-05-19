# Local-LLM-DeepSeek-Memory — Demo Persistent Memory Terminal

**Repository:** [https://github.com/mdwoicke/Local-LLM-DeepSeek-Memory](https://github.com/mdwoicke/Local-LLM-DeepSeek-Memory)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐ (Thấp, chỉ lấy ý tưởng nhỏ)

## Tổng quan

Đây là một dự án demo nhỏ (2 stars, 0 forks) — một terminal chat Python sử dụng Ollama với DeepSeek model và persistent memory dùng file `.pt` (PyTorch tensor). Dù đơn giản, nó chứa một vài ý tưởng thú vị có thể áp dụng như cơ chế fallback.

## Tính năng nổi bật

### 1. Fallback Embedding bằng SHA-256
Khi LLM embedding service không khả dụng, hệ thống tự động chuyển sang dùng SHA-256 hash của văn bản làm deterministic vector. Dù thô sơ, điều này đảm bảo hệ thống không bị gián đoạn.
→ *Áp dụng:* MemoryMesh có thể áp dụng cơ chế tương tự: nếu 9Router hoặc embedding service lỗi, dùng local hash làm vector tạm thời để duy trì search cơ bản. Sau đó đồng bộ lại khi service khả dụng.

### 2. Fixed-Size Memory với Usage Vector
- Bộ nhớ có kích thước cố định (fixed number of slots).
- Mỗi slot có một "usage vector" để quyết định mức độ quan trọng và cần update hay không.
→ *Áp dụng:* Khi MemoryMesh cần giới hạn số lượng ký ức lưu trữ (để kiểm soát dung lượng), có thể dùng cơ chế đánh trọng số và eviction dựa trên usage vector (tần suất truy cập, độ mới).

### 3. Đơn giản, dễ hiểu
Toàn bộ code rất ngắn, dễ đọc, dễ thử nghiệm. Có thể dùng để prototype nhanh ý tưởng trước khi tích hợp vào MemoryMesh.

## Hạn chế

- Quá sơ khai, không có khả năng mở rộng.
- Không có MCP integration, không có retrieval pipeline.
- Lưu vector dạng file `.pt` không hiệu quả, khó truy vấn.

## Kết luận

Repo này chỉ có giá trị tham khảo ở cấp độ ý tưởng (fallback hash, fixed-size eviction). Không nên dùng làm nền tảng cho MemoryMesh.