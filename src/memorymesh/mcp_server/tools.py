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
                "level": {
                    "type": "string",
                    "enum": ["user", "session", "knowledge"],
                    "description": "Cấp độ ký ức: user (thông tin cá nhân), session (ngữ cảnh hội thoại), knowledge (kiến thức chung). Mặc định user.",
                },
                "workspace_path": {"type": "string", "description": "Đường dẫn workspace để giới hạn phạm vi (mặc định từ session hiện tại)"},
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
                "workspace_path": {"type": "string", "description": "Giới hạn recall theo workspace (mặc định session hiện tại)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="forget",
        description="Xóa (soft-delete) một ký ức theo ID. Ký ức sẽ không còn xuất hiện trong recall/list, nhưng có thể khôi phục bằng unarchive_memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID của ký ức cần xóa"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="archive_memory",
        description="Chuyển một ký ức vào kho lưu trữ. Ký ức sẽ không xuất hiện trong recall/list cho đến khi được unarchive.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID của ký ức cần archive"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="unarchive_memory",
        description="Khôi phục một ký ức đã archive. Ký ức sẽ xuất hiện trở lại trong recall/list.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID của ký ức cần unarchive"},
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
    Tool(
        name="save_system_prompt",
        description="Lưu system prompt của phiên làm việc hiện tại.",
        inputSchema={
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt cần lưu"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
            "required": ["system_prompt"],
        },
    ),
    Tool(
        name="save_context_pair",
        description="Lưu một cặp hội thoại (user message + assistant response) vào session.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_message": {"type": "string", "description": "Tin nhắn của user"},
                "assistant_message": {"type": "string", "description": "Phản hồi của assistant"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
            "required": ["user_message", "assistant_message"],
        },
    ),
    Tool(
        name="list_sessions",
        description="Liệt kê các session đã lưu.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Số lượng kết quả (mặc định 10)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
        },
    ),
    Tool(
        name="get_session_context",
        description="Lấy toàn bộ context của một session cũ (system prompt + hội thoại).",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID của session cần xem"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Số dòng context tối đa (mặc định 50)"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="new_session",
        description="Tạo một session làm việc mới. Session cũ sẽ tự động được đóng lại.",
        inputSchema={
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt cho session mới"},
                "workspace_path": {"type": "string", "description": "Đường dẫn workspace"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
        },
    ),
    Tool(
        name="end_session",
        description="Kết thúc session làm việc hiện tại.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID session cần kết thúc (mặc định session hiện tại)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
        },
    ),
    Tool(
        name="save_workspace_context",
        description="Chụp nhanh trạng thái workspace hiện tại (danh sách file, git status, dependencies).",
        inputSchema={
            "type": "object",
            "properties": {
                "workspace_path": {"type": "string", "description": "Đường dẫn workspace để snapshot (mặc định từ session)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
        },
    ),
    Tool(
        name="resume_session",
        description="Khôi phục context của một session cũ vào session hiện tại. Recall memories từ session cũ.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID session cần khôi phục"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Số memory cần recall (mặc định 10)"},
                "user_id": {"type": "string", "description": "ID người dùng"},
            },
            "required": ["session_id"],
        },
    ),
]