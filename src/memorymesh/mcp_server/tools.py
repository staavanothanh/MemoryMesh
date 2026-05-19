from mcp.types import Tool

TOOLS = [
    Tool(
        name="remember",
        description="Lưu một ký ức mới. Hệ thống sẽ tự động phân tích và làm giàu metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Nội dung cần ghi nhớ"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách tag (tùy chọn)",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Mức độ quan trọng 1-5 (mặc định 3)",
                },
                "user_id": {"type": "string", "description": "ID người dùng (mặc định từ config)"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="recall",
        description="Truy xuất những ký ức liên quan nhất đến một truy vấn.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu truy vấn để tìm ký ức"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Số lượng kết quả tối đa (mặc định 5)",
                },
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="forget",
        description="Xóa một ký ức theo ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID của ký ức cần xóa"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="list_memories",
        description="Liệt kê tất cả ký ức của người dùng, có phân trang.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Số lượng kết quả (mặc định 100)"},
                "offset": {"type": "integer", "minimum": 0, "description": "Vị trí bắt đầu (mặc định 0)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
        },
    ),
    Tool(
        name="ping",
        description="Kiểm tra server còn sống.",
        inputSchema={"type": "object", "properties": {}},
    ),
]