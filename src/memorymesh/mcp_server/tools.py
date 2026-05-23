from mcp.types import Tool

TOOLS = [
    Tool(
        name="remember",
        description="Save important info (imp=5). Auto-enriches. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to remember"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags (optional)",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Importance level 1-5 (default 3)",
                },
                "level": {
                    "type": "string",
                    "enum": ["user", "session", "knowledge"],
                    "description": "Memory level: user (personal info), session (conversation context), knowledge (general knowledge). Default user.",
                },
                "workspace_path": {"type": "string", "description": "Workspace path to limit scope (default from current session)"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="recall",
        description="Recall memories. Call after new_session. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query string to find memories"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of results (default 5)",
                },
                "workspace_path": {"type": "string", "description": "Limit recall by workspace (default current session)"}
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="forget",
        description="Soft-delete a memory by ID. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID of the memory to delete"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="archive_memory",
        description="Archive a memory (soft delete). |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID of the memory to archive"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="unarchive_memory",
        description="Restore an archived memory. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID of the memory to unarchive"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="list_memories",
        description="List user memories (paginated). |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Number of results (default 100)"},
                "offset": {"type": "integer", "minimum": 0, "description": "Starting position (default 0)"}
            },
        },
    ),
    Tool(
        name="ping",
        description="Health check. |→save_context_pair",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="save_system_prompt",
        description="Save system prompt for session. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt to save"}
            },
            "required": ["system_prompt"],
        },
    ),
    Tool(
        name="save_context_pair",
        description="⚠️ MANDATORY after EVERY response. Save user+assistant. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "user_message": {"type": "string", "description": "User's message"},
                "assistant_message": {"type": "string", "description": "Assistant's response"}
            },
            "required": ["user_message", "assistant_message"],
        },
    ),
    Tool(
        name="list_sessions",
        description="List saved sessions. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results (default 10)"}
            },
        },
    ),
    Tool(
        name="get_session_context",
        description="Get full context of old session. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID of the session to view"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Maximum context lines (default 50)"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="new_session",
        description="⚠️ CALL FIRST. Create new session, close old. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt for the new session"},
                "workspace_path": {"type": "string", "description": "Workspace path"}
            },
        },
    ),
    Tool(
        name="end_session",
        description="End session, compress, flush buffer. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to end (default current session)"}
            },
        },
    ),
    Tool(
        name="save_workspace_context",
        description="Snapshot workspace state. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "workspace_path": {"type": "string", "description": "Workspace path for snapshot (default from session)"}
            },
        },
    ),
    Tool(
        name="delete_session",
        description="Delete session + its memories. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to delete (default current session)"}
            },
        },
    ),
    Tool(
        name="preserve_session_memories",
        description="Scan session, promote key memories. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to scan (default current session)"}
            },
        },
    ),
    Tool(
        name="resume_session",
        description="Restore old session context. |→save_context_pair",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to restore"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Number of memories to recall (default 10)"}
            },
            "required": ["session_id"],
        },
    ),
]