# LightAgent — Lightweight Agent Framework (Mem0 + MCP)

**Repository:** [https://github.com/wanxingai/LightAgent](https://github.com/wanxingai/LightAgent)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐⭐⭐ (Rất cao)

## Tổng quan

LightAgent là một framework xây dựng agent nhẹ, tích hợp sẵn **mem0** làm memory backend, hỗ trợ MCP protocol (stdio/SSE), và tương thích với DeepSeek. Nó là một ví dụ tham khảo tuyệt vời về cách "bọc" một công cụ memory (mem0) vào một agent có khả năng gọi tool.

## Tính năng nổi bật

### 1. Tích hợp mem0 làm Memory Backend
- Tự động quản lý trí nhớ người dùng trong suốt hội thoại.
- Cấu hình đơn giản, chỉ cần set provider (DeepSeek, OpenAI, v.v.).
→ *Áp dụng:* Cách LightAgent khởi tạo và gọi mem0 là tài liệu mẫu cho việc tích hợp mem0 vào MemoryMesh (nếu chọn mem0 làm engine).

### 2. Hệ thống Tool được thiết kế rõ ràng
Sử dụng decorator `@tool_info` để định nghĩa tool, bao gồm mô tả, tham số.
```python
@tool_info(
    name="remember",
    description="Store important information",
    parameters={...}
)
def remember(self, content: str) -> str:
    ...
```
→ Áp dụng: MemoryMesh có thể áp dụng mẫu thiết kế này để định nghĩa các MCP tool remember, recall, forget, tag một cách rõ ràng, dễ bảo trì.

### 3. Hỗ trợ MCP Protocol (stdio/SSE)
Có thể chạy như một MCP server, giao tiếp qua stdio hoặc HTTP SSE.

Đã kiểm chứng tương thích với Claude Desktop và Cursor.
→ Áp dụng: Tham khảo code phần MCP server để xây dựng giao tiếp giữa DeepSeek-TUI và MemoryMesh.

### 4. Tree-of-Thought Reasoning
Hỗ trợ suy luận dạng cây (ToT) cho các tác vụ phức tạp, có thể kết hợp với memory để lưu trữ các nhánh suy luận.
→ Áp dụng: Trong tương lai, MemoryMesh có thể lưu cả "cây suy luận" vào bộ nhớ để tái sử dụng khi gặp vấn đề tương tự (mở rộng từ Experience memory của sage-memory).

### Hướng áp dụng vào MemoryMesh
- Tham khảo cách tổ chức MCP server: Đặc biệt là cách đăng ký tool, xử lý request/response theo chuẩn MCP.

- Mẫu tích hợp mem0: Nếu dùng mem0 làm backend, có thể tham khảo cách LightAgent khởi tạo và gọi các phương thức add/search.

- Thiết kế tool tương lai: Áp dụng mô hình tool_info để thêm các tool mới như summarize_session, get_user_profile.