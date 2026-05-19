# sage-memory — Self-Learning MCP Memory Server

**Repository:** [https://github.com/xoai/sage-memory](https://github.com/xoai/sage-memory)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐⭐⭐⭐ (Cực kỳ cao)

## Tổng quan

Sage-memory là một MCP server memory được thiết kế với triết lý "học" (learn) chứ không chỉ "nhớ" (remember). Nó sử dụng duy nhất một file SQLite cho toàn bộ dữ liệu và vector, không yêu cầu hạ tầng phức tạp. Đây là nguồn tham khảo kiến trúc xuất sắc nhất cho MemoryMesh.

## Tính năng nổi bật

### 1. Ba loại trí nhớ trong một lần truy xuất
- **Knowledge:** Hiểu biết, sự kiện (factual).
- **Structure:** Mối quan hệ giữa các thực thể.
- **Experience:** Bài học từ lỗi sai, quy tắc phòng tránh.
→ *Áp dụng:* MemoryMesh có thể mở rộng từ "nhớ sự kiện" sang "hiểu ngữ cảnh" và "học từ sai lầm" bằng cách phân loại tương tự.

### 2. Self-Learning Loop (Vòng lặp tự học)
```
mistake → prevention rule → recall → improvement
```
Khi phát hiện mâu thuẫn giữa ký ức mới và cũ, hệ thống tự động tạo rule để tránh lặp lại sai lầm.
→ *Áp dụng:* Tính năng quản lý trùng lặp / mâu thuẫn trong Giai đoạn 4 của MemoryMesh sẽ dựa trên mô hình này. Thay vì chỉ merge, MemoryMesh có thể sinh rule phòng ngừa.

### 3. Retrieval Pipeline 6 giai đoạn
1. **Expand** — Mở rộng query thành nhiều biến thể.
2. **Retrieve** — Tìm kiếm song song qua 3 channel (BM25, vector, entity graph).
3. **Fuse** — Kết hợp điểm số từ các channel.
4. **Dedup** — Loại bỏ kết quả trùng lặp.
5. **Rerank** — Xếp hạng lại bằng mô hình điểm tinh chỉnh.
6. **Score** — Trả về kết quả cuối cùng với độ tin cậy.
→ *Áp dụng:* Đây là bản thiết kế hoàn chỉnh cho tầng `retrieval` trong MemoryMesh. Có thể triển khai từng bước, bắt đầu từ Retrieve và Fuse.

### 4. Zero Infrastructure — Một file SQLite duy nhất
- Dùng extension `sqlite-vec` để lưu vector embedding ngay trong SQLite (không cần ChromaDB).
- FTS5 (Full-Text Search) cho BM25.
- Tất cả chỉ ~3,000 dòng code Python, không Docker, không Redis.
→ *Áp dụng:* MemoryMesh có thể chọn giải pháp siêu nhẹ này thay vì ChromaDB, giúp đóng gói Docker chỉ vài MB, dễ triển khai mọi nơi.

### 5. MCP Native
Cấu hình trực tiếp trong `mcp_servers.json` cho Claude Code, Cursor, hoặc bất kỳ MCP client nào.
```json
{
  "mcpServers": {
    "sage": {
      "command": "python",
      "args": ["path/to/server.py"]
    }
  }
}
```
→ *Áp dụng:* Mẫu cấu hình này sẽ được dùng để tích hợp MemoryMesh vào DeepSeek-TUI.

### Hướng áp dụng vào MemoryMesh
- Áp dụng retrieval pipeline 6 giai đoạn: Đây sẽ là xương sống của search_memory(). Khởi đầu bằng việc kết hợp BM25 + vector, sau đó thêm entity graph.

- Sử dụng SQLite + sqlite-vec thay ChromaDB: Giảm phụ thuộc, tối giản hóa stack kỹ thuật, phù hợp với yêu cầu local-first.

- Tích hợp Self-Learning Loop: Khi phát hiện trùng lặp (cosine similarity > threshold), tự động tạo bản ghi experience để hệ thống không lặp lại xung đột.

- Học cách tổ chức MCP server: Tham khảo code server.py của sage-memory để viết server MCP chuẩn cho MemoryMesh.