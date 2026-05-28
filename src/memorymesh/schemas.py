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
import json


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

class BaseToolInput(BaseModel, extra="forbid"):
    """Base class for all MCP tool inputs — ensures user_id is never silently dropped."""
    user_id: str = Field(default="", description="User ID (defaults to configured user)")


class RecallCursor(BaseModel):
    """Cursor for paginated recall — encodes search state for consistent pagination."""
    query_hash: str = Field(..., description="sha256 of query[:16] for cursor validation")
    tier: str = Field(..., pattern=r"^(semantic|fts_keyword|chronological)$")
    last_score: float = Field(..., ge=0.0, le=1.0)
    last_id: str = Field(..., min_length=1)
    page: int = Field(default=1, ge=1)


class RememberInput(BaseToolInput):
    """Input for the remember tool — save a memory."""
    content: str = Field(..., min_length=1, description="Content to remember")
    tags: List[str] = Field(default_factory=list, description="List of tags (optional)")
    importance: int = Field(default=3, ge=1, le=5, description="Importance level 1-5")
    level: Literal["user", "session", "knowledge"] = Field(
        default="user",
        description="Memory level: user (personal info), session (conversation context), knowledge (general knowledge)"
    )
    workspace_path: str = Field(default="", description="Workspace path to limit scope")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class RecallInput(BaseToolInput):
    """Input for the recall tool — search memories."""
    query: str = Field(..., min_length=1, description="Query string to find memories")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    workspace_path: str = Field(default="", description="Limit recall by workspace")
    max_tokens: Optional[int] = Field(default=None, description="Token budget for results")
    cursor: Optional[str] = Field(default=None, description="JSON cursor for pagination: {'last_score': 0.85, 'last_id': '...', 'page': 2}. Omit for first page.")

    @field_validator("cursor")
    @classmethod
    def cursor_must_be_valid_json(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                json.loads(v)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("cursor must be a valid JSON string")
        return v


class ForgetInput(BaseToolInput):
    """Input for the forget tool — soft-delete a memory."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to delete")


class ArchiveMemoryInput(BaseToolInput):
    """Input for the archive_memory tool."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to archive")


class UnarchiveMemoryInput(BaseToolInput):
    """Input for the unarchive_memory tool."""
    memory_id: str = Field(..., min_length=1, description="ID of the memory to unarchive")


class ListMemoriesInput(BaseToolInput):
    """Input for the list_memories tool — paginated listing."""
    limit: int = Field(default=100, ge=1, le=1000, description="Number of results")
    offset: int = Field(default=0, ge=0, description="Starting position")


class PingInput(BaseToolInput):
    """Input for the ping tool — health check (no required fields)."""


class SaveSystemPromptInput(BaseToolInput):
    """Input for the save_system_prompt tool."""
    system_prompt: str = Field(..., min_length=1, description="System prompt to save")


class CommitMilestoneInput(BaseToolInput):
    """Input for the commit_milestone tool."""
    summary: str = Field(..., min_length=1, description="Brief summary of what was accomplished")
    tasks_done: str = Field(default="", description="Files changed, bugs fixed, or features completed")
    next_steps: str = Field(default="", description="Planned next steps")


class SaveContextPairInput(BaseToolInput):
    """Input for the save_context_pair tool (deprecated)."""
    user_message: str = Field(..., min_length=1, description="User's message")
    assistant_message: str = Field(default="", description="Assistant's response")


class ListSessionsInput(BaseToolInput):
    """Input for the list_sessions tool."""
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")


class GetSessionContextInput(BaseToolInput):
    """Input for the get_session_context tool."""
    session_id: str = Field(..., min_length=1, description="ID of the session to view")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum context lines")


class NewSessionInput(BaseToolInput):
    """Input for the new_session tool."""
    system_prompt: str = Field(default="", description="System prompt for the new session")
    workspace_path: str = Field(default="", description="Workspace path")


class EndSessionInput(BaseToolInput):
    """Input for the end_session tool."""
    session_id: str = Field(default="", description="Session ID to end (default current)")


class SaveWorkspaceContextInput(BaseToolInput):
    """Input for the save_workspace_context tool."""
    workspace_path: str = Field(default="", description="Workspace path for snapshot")


class DeleteSessionInput(BaseToolInput):
    """Input for the delete_session tool."""
    session_id: str = Field(default="", description="Session ID to delete")


class PreserveSessionMemoriesInput(BaseToolInput):
    """Input for the preserve_session_memories tool."""
    session_id: str = Field(default="", description="Session ID to scan")


class ResumeSessionInput(BaseToolInput):
    """Input for the resume_session tool."""
    session_id: str = Field(..., min_length=1, description="Session ID to restore")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of memories to recall")


# ── Knowledge Graph Tool Input Models ───────────────────────────────────

_KNOWN_ENTITY_TYPES = frozenset({
    "concept", "project", "module", "bug", "person", "tool", "file", "function", "class",
})

class CreateEntityInput(BaseToolInput):
    """Input for the create_entity tool — a cognitive operation that records a concept/entity into the Knowledge Graph.
    Safe for Read-Only/Plan Mode: this operation does NOT modify the file system or execute external commands.
    It only records the model's own reasoning and conclusions for future recall via the Knowledge Graph.
    """
    name: str = Field(..., min_length=1, max_length=200, description="Name of the entity (concept, project, module, bug, etc.)")
    entity_type: str = Field(default="concept", description="Type of entity: concept, project, module, bug, person, tool, etc.")
    properties: Optional[str] = Field(default=None, description="Optional JSON string of entity properties/metadata")
    workspace_path: str = Field(default="", description="Workspace path to limit scope")

    @field_validator("entity_type")
    @classmethod
    def entity_type_must_be_known(cls, v: str) -> str:
        if v not in _KNOWN_ENTITY_TYPES:
            raise ValueError(f"Unknown entity_type: {v}. Must be one of {sorted(_KNOWN_ENTITY_TYPES)}")
        return v

    @field_validator("properties")
    @classmethod
    def properties_must_be_valid_json(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                json.loads(v)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("properties must be a valid JSON string")
        return v

_KNOWN_RELATION_TYPES = frozenset({
    "SOLVES", "DEPENDS_ON", "IMPLEMENTS", "USES", "HAS_PREFERENCE",
    "RELATES_TO", "EXTENDS", "CONTAINS", "CALLS", "CREATES",
})

class CreateRelationInput(BaseToolInput):
    """Input for the create_relation tool — a cognitive operation that links two entities in the Knowledge Graph.
    Safe for Read-Only/Plan Mode: this operation does NOT modify the file system or execute external commands.
    It only records the model's own reasoning about how concepts relate to each other.
    """
    source: str = Field(..., min_length=1, max_length=200, description="Name of the source entity")
    target: str = Field(..., min_length=1, max_length=200, description="Name of the target entity")
    relation_type: str = Field(..., min_length=1, max_length=100, description="Type of relation: SOLVES, DEPENDS_ON, IMPLEMENTS, USES, HAS_PREFERENCE, etc.")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relation weight/relevance (0.0-1.0)")
    workspace_path: str = Field(default="", description="Workspace path to limit scope")

    @field_validator("relation_type")
    @classmethod
    def relation_type_must_be_known(cls, v: str) -> str:
        if v not in _KNOWN_RELATION_TYPES:
            raise ValueError(f"Unknown relation_type: {v}. Must be one of {sorted(_KNOWN_RELATION_TYPES)}")
        return v

class QueryGraphInput(BaseToolInput):
    """Input for the query_graph tool — find 1-hop neighbors of an entity in the Knowledge Graph."""
    entity_name: str = Field(..., min_length=1, description="Name of the entity to query")
    limit: int = Field(default=20, ge=1, le=50, description="Maximum number of relations to return")
    workspace_path: str = Field(default="", description="Workspace path to limit scope")


class TraceEntityInput(BaseToolInput):
    """Input for the trace_entity tool — traverse multi-hop paths from an entity in the Knowledge Graph."""
    entity_name: str = Field(..., min_length=1, description="Name of the entity to start tracing from")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum traversal depth")
    max_relations: int = Field(default=20, ge=1, le=30, description="Maximum number of relations to return")
    workspace_path: str = Field(default="", description="Workspace path to limit scope")


class RecallRawInput(BaseToolInput):
    """Input for the recall_raw tool — query raw tool call history."""
    session_id: str = Field(default="", description="Session ID (default current session)")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of entries")
    offset: int = Field(default=0, ge=0, description="Starting position")
    tool_name: str = Field(default="", description="Filter by tool name")
    status: Optional[Literal["success", "error"]] = Field(default=None, description="Filter by status (omit for no filter)")

class LearnSessionInput(BaseToolInput):
    """Input for the learn_session tool — analyze a session and extract behavioral patterns."""
    session_id: str = Field(default="", description="Session ID to learn from (default current)")

class MergeEntitiesInput(BaseToolInput):
    """Input for the merge_entities tool — merge two knowledge graph entities."""
    source: str = Field(..., min_length=1, max_length=200, description="Name of the source entity (will be merged into target)")
    target: str = Field(..., min_length=1, max_length=200, description="Name of the target entity (will absorb source)")
    user_id: str = Field(default="", description="User ID (defaults to configured user)")

    @field_validator("source")
    @classmethod
    def source_not_equal_target(cls, v: str, info) -> str:
        if "target" in info.data and v.strip().lower() == info.data["target"].strip().lower():
            raise ValueError("source and target must be different entities")
        return v


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
    "create_entity": CreateEntityInput,
    "create_relation": CreateRelationInput,
    "query_graph": QueryGraphInput,
    "trace_entity": TraceEntityInput,
    "recall_raw": RecallRawInput,
    "learn_session": LearnSessionInput,
    "merge_entities": MergeEntitiesInput,
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
