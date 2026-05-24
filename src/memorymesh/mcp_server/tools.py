from mcp.types import Tool

MUTATE_TOOLS = frozenset({
    "forget", "archive_memory", "unarchive_memory", "save_system_prompt",
    "new_session", "end_session", "delete_session", "preserve_session_memories",
    "save_workspace_context", "resume_session",
})

TOOLS = [
    Tool(
        name="remember",
        description="Save important info (imp=5). Auto-enriches.",
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
        description="Recall memories by query. Call after new_session.",
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
        description="Soft-delete a memory by ID.",
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
        description="Archive a memory (soft delete).",
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
        description="Restore an archived memory.",
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
        description="List user memories (paginated).",
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
        description="Health check.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="save_system_prompt",
        description="Save system prompt for session.",
        inputSchema={
            "type": "object",
            "properties": {
                "system_prompt": {"type": "string", "description": "System prompt to save"}
            },
            "required": ["system_prompt"],
        },
    ),
    Tool(
        name="commit_milestone",
        description="Commit a milestone: summarize completed work, current state, and next steps. Call ONLY when finishing a logical block of work (e.g., multiple edits, a feature, a bug fix). Do NOT call after every response. Releases any held search results.",
        inputSchema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary (1-2 sentences) of what was accomplished"},
                "tasks_done": {"type": "string", "description": "Files changed, bugs fixed, or features completed"},
                "next_steps": {"type": "string", "description": "Planned next steps — what to work on next"}
            },
            "required": ["summary", "tasks_done", "next_steps"],
        },
    ),
    Tool(
        name="save_context_pair",
        description="DEPRECATED — use commit_milestone instead. Saves a user+assistant exchange (auto-mapped to commit_milestone).",
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
        description="List saved sessions.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results (default 10)"}
            },
        },
    ),
    Tool(
        name="get_session_context",
        description="Get full context of old session.",
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
        description="Create a new session. Call once at conversation start. After calling, call commit_milestone when logical work blocks complete. Your text responses are not auto-captured.",
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
        description="End session, compress, flush buffer.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to end (default current session)"}
            },
        },
    ),
    Tool(
        name="save_workspace_context",
        description="Snapshot workspace state.",
        inputSchema={
            "type": "object",
            "properties": {
                "workspace_path": {"type": "string", "description": "Workspace path for snapshot (default from session)"}
            },
        },
    ),
    Tool(
        name="delete_session",
        description="Delete session + its memories.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to delete (default current session)"}
            },
        },
    ),
    Tool(
        name="preserve_session_memories",
        description="Scan session, promote key memories.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to scan (default current session)"}
            },
        },
    ),
    Tool(
        name="resume_session",
        description="Restore old session context. Auto-called when resuming previous work.",
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