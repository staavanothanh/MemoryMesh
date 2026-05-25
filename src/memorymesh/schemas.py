"""Pydantic models for MemoryMesh MCP tool inputs, outputs, and domain types.

All MCP tool arguments are validated against these models before reaching
the handler functions. This ensures runtime type safety beyond the JSON
Schema declared in tools.py.

JSON Schema for MCP tool registration is auto-generated via model_json_schema().

Internal domain models (MemoryRecord, SearchResult) remain as TypedDict for
backward compatibility with dict-access patterns throughout the codebase.
"""

from __future__ import annotations

from typing import Optional, List, Union, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


# ── Domain Models (TypedDict for dict-access compatibility) ──────────────

class MemoryRecord(TypedDict):
    """Full memory record as stored in the backend."""
    id: str
    user_id: str
    content: str
    tags: List[str]
    importance: int
    timestamp: str
    level: str
    workspace_path: str


class SearchResult(TypedDict):
    """Search result returned to the MCP client."""
    id: str
    content: str
    score: float
    tags: List[str]
    importance: int
    timestamp: str


class ToolOutput(TypedDict, total=False):
    """Standard MCP tool response envelope."""
    status: str  # "success" or "error"
    data: Union[dict, list, str, None]
    error: Optional[str]
    formatted: Optional[str]
    meta: Optional[dict]


# ── MCP Tool Input Models ───────────────────────────────────────────────

class RememberInput(BaseModel):
    """Input for the remember tool — save a memory."""
    content: str = Field(..., min_length=1, description="Content to remember")
    tags: List[str] = Field(default_factory=list, description="List of tags (optional)")
    importance: int = Field(default=3, ge=1, le=5, description="Importance level 1-5")
    level: Literal["user", "session", "knowledge"] = Field(
        default="user",
        description="Memory level: user (personal info), session (conversation context), knowledge (general knowledge)"
    )
    workspace_path: str = Field(default="", description="Workspace path to limit scope")
    user_id: Optional[str] = Field(default=None, description="User ID (default from config)")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class RecallInput(BaseModel):
    """Input for the recall tool — search memories."""
    query: str = Field(..., min_length=1, description="Query string to find memories")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    workspace_path: str = Field(default="", description="Limit recall by workspace")
    user_id: Optional[str] = Field(default=None, description="User ID (default from config)")
    max_tokens: Optional[int] = Field(default=None, description="Token budget for results")


class ForgetInput(BaseModel):
    """Input for the forget tool — soft-delete a memory."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to delete")


class ArchiveMemoryInput(BaseModel):
    """Input for the archive_memory tool."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to archive")


class UnarchiveMemoryInput(BaseModel):
    """Input for the unarchive_memory tool."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to unarchive")


class ListMemoriesInput(BaseModel):
    """Input for the list_memories tool — paginated listing."""
    limit: int = Field(default=100, ge=1, le=1000, description="Number of results")
    offset: int = Field(default=0, ge=0, description="Starting position")
    user_id: Optional[str] = Field(default=None, description="User ID (default from config)")


class PingInput(BaseModel):
    """Input for the ping tool — health check (no required fields)."""
    user_id: Optional[str] = Field(default=None)


class SaveSystemPromptInput(BaseModel):
    """Input for the save_system_prompt tool."""
    system_prompt: str = Field(..., min_length=1, description="System prompt to save")
    user_id: Optional[str] = Field(default=None)


class CommitMilestoneInput(BaseModel):
    """Input for the commit_milestone tool."""
    summary: str = Field(..., min_length=1, description="Brief summary of what was accomplished")
    tasks_done: str = Field(default="", description="Files changed, bugs fixed, or features completed")
    next_steps: str = Field(default="", description="Planned next steps")
    user_id: Optional[str] = Field(default=None)


class SaveContextPairInput(BaseModel):
    """Input for the save_context_pair tool (deprecated)."""
    user_message: str = Field(..., min_length=1, description="User's message")
    assistant_message: str = Field(default="", description="Assistant's response")


class ListSessionsInput(BaseModel):
    """Input for the list_sessions tool."""
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    user_id: Optional[str] = Field(default=None)


class GetSessionContextInput(BaseModel):
    """Input for the get_session_context tool."""
    session_id: str = Field(..., min_length=1, description="ID of the session to view")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum context lines")


class NewSessionInput(BaseModel):
    """Input for the new_session tool."""
    system_prompt: str = Field(default="", description="System prompt for the new session")
    workspace_path: str = Field(default="", description="Workspace path")
    user_id: Optional[str] = Field(default=None)


class EndSessionInput(BaseModel):
    """Input for the end_session tool."""
    session_id: str = Field(default="", description="Session ID to end (default current)")


class SaveWorkspaceContextInput(BaseModel):
    """Input for the save_workspace_context tool."""
    workspace_path: str = Field(default="", description="Workspace path for snapshot")


class DeleteSessionInput(BaseModel):
    """Input for the delete_session tool."""
    session_id: str = Field(default="", description="Session ID to delete")


class PreserveSessionMemoriesInput(BaseModel):
    """Input for the preserve_session_memories tool."""
    session_id: str = Field(default="", description="Session ID to scan")


class ResumeSessionInput(BaseModel):
    """Input for the resume_session tool."""
    session_id: str = Field(..., min_length=1, description="Session ID to restore")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of memories to recall")


# ── Mapping: tool name → Pydantic model ─────────────────────────────────

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "remember": RememberInput,
    "recall": RecallInput,
    "forget": ForgetInput,
    "archive_memory": ArchiveMemoryInput,
    "unarchive_memory": UnarchiveMemoryInput,
    "list_memories": ListMemoriesInput,
    "ping": PingInput,
    "save_system_prompt": SaveSystemPromptInput,
    "commit_milestone": CommitMilestoneInput,
    "save_context_pair": SaveContextPairInput,
    "list_sessions": ListSessionsInput,
    "get_session_context": GetSessionContextInput,
    "new_session": NewSessionInput,
    "end_session": EndSessionInput,
    "save_workspace_context": SaveWorkspaceContextInput,
    "delete_session": DeleteSessionInput,
    "preserve_session_memories": PreserveSessionMemoriesInput,
    "resume_session": ResumeSessionInput,
}


def validate_tool_input(tool_name: str, args: dict) -> BaseModel:
    """Validate raw MCP tool arguments against the corresponding Pydantic model.

    Raises pydantic.ValidationError if validation fails.
    """
    model = TOOL_INPUT_MODELS.get(tool_name)
    if model is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return model(**args)


# ── Auto-generated JSON Schemas for MCP Tool Registration ───────────────

def generate_tool_json_schema(tool_name: str) -> dict:
    """Generate a JSON Schema for MCP tool registration from the Pydantic model."""
    model = TOOL_INPUT_MODELS.get(tool_name)
    if model is None:
        return {"type": "object", "properties": {}}

    schema = model.model_json_schema()
    # Remove pydantic-specific fields not needed by MCP
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema
