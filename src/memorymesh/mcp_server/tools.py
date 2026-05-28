"""MCP tool definitions with auto-generated JSON schemas from Pydantic models."""

from mcp.types import Tool

from ..schemas import (
    RememberInput,
    RecallInput,
    ForgetInput,
    ArchiveMemoryInput,
    UnarchiveMemoryInput,
    ListMemoriesInput,
    PingInput,
    SaveSystemPromptInput,
    CommitMilestoneInput,
    SaveContextPairInput,
    ListSessionsInput,
    GetSessionContextInput,
    NewSessionInput,
    EndSessionInput,
    SaveWorkspaceContextInput,
    DeleteSessionInput,
    PreserveSessionMemoriesInput,
    ResumeSessionInput,
    CreateEntityInput,
    CreateRelationInput,
    QueryGraphInput,
    TraceEntityInput,
    RecallRawInput,
    LearnSessionInput,
    MergeEntitiesInput,
)


def _schema(model: type) -> dict:
    """Extract JSON Schema from a Pydantic model, stripping pydantic internals."""
    s = model.model_json_schema()
    s.pop("title", None)
    for prop in s.get("properties", {}).values():
        prop.pop("title", None)
    return s


MUTATE_TOOLS = frozenset({
    "forget", "archive_memory", "unarchive_memory", "save_system_prompt",
    "new_session", "end_session", "delete_session", "preserve_session_memories",
    "save_workspace_context", "resume_session",
})

COGNITIVE_TOOLS = frozenset({
    "create_entity", "create_relation", "query_graph", "trace_entity",
})

EXPENSIVE_TOOLS = frozenset({"recall", "save_workspace_context", "remember"})

TOOLS = [
    Tool(
        name="remember",
        description="Save important info (imp=5). Auto-enriches.",
        inputSchema=_schema(RememberInput),
    ),
    Tool(
        name="recall",
        description="Recall memories by query. Response includes context_restored signal in meta — true when bootstrap context loaded. Call this FIRST on session start.",
        inputSchema=_schema(RecallInput),
    ),
    Tool(
        name="forget",
        description="Soft-delete a memory by ID.",
        inputSchema=_schema(ForgetInput),
    ),
    Tool(
        name="archive_memory",
        description="Archive a memory (soft delete).",
        inputSchema=_schema(ArchiveMemoryInput),
    ),
    Tool(
        name="unarchive_memory",
        description="Restore an archived memory.",
        inputSchema=_schema(UnarchiveMemoryInput),
    ),
    Tool(
        name="list_memories",
        description="List user memories (paginated).",
        inputSchema=_schema(ListMemoriesInput),
    ),
    Tool(
        name="ping",
        description="Health check.",
        inputSchema=_schema(PingInput),
    ),
    Tool(
        name="save_system_prompt",
        description="Save system prompt for session.",
        inputSchema=_schema(SaveSystemPromptInput),
    ),
    Tool(
        name="commit_milestone",
        description="Commit a milestone: summarize completed work, current state, and next steps. Call ONLY when finishing a logical block of work (e.g., multiple edits, a feature, a bug fix). Milestone saves are for checkpoints, not per-response. Releases any held search results.",
        inputSchema=_schema(CommitMilestoneInput),
    ),
    Tool(
        name="save_context_pair",
        description="DEPRECATED — use commit_milestone instead. Saves a user+assistant exchange (auto-mapped to commit_milestone).",
        inputSchema=_schema(SaveContextPairInput),
    ),
    Tool(
        name="list_sessions",
        description="List saved sessions.",
        inputSchema=_schema(ListSessionsInput),
    ),
    Tool(
        name="get_session_context",
        description="Get full context of old session.",
        inputSchema=_schema(GetSessionContextInput),
    ),
    Tool(
        name="new_session",
        description="Create a new session. Call once at conversation start. After calling, call commit_milestone when logical work blocks complete. Your text responses are not auto-captured.",
        inputSchema=_schema(NewSessionInput),
    ),
    Tool(
        name="end_session",
        description="End session, compress, flush buffer.",
        inputSchema=_schema(EndSessionInput),
    ),
    Tool(
        name="save_workspace_context",
        description="Snapshot workspace state.",
        inputSchema=_schema(SaveWorkspaceContextInput),
    ),
    Tool(
        name="delete_session",
        description="Delete session + its memories.",
        inputSchema=_schema(DeleteSessionInput),
    ),
    Tool(
        name="preserve_session_memories",
        description="Scan session, promote key memories.",
        inputSchema=_schema(PreserveSessionMemoriesInput),
    ),
    Tool(
        name="resume_session",
        description="Restore old session context. Auto-called when resuming previous work.",
        inputSchema=_schema(ResumeSessionInput),
    ),

    # ── Knowledge Graph Tools (Cognitive Operations — safe for Plan/Read-Only) ──────

    Tool(
        name="create_entity",
        description="[COGNITIVE OPERATION — SAFE FOR PLAN/READ-ONLY MODE] Create a knowledge entity in the Knowledge Graph. Does NOT modify file system. Records the model's concepts, conclusions, plans, and reasoning as structured graph nodes for future recall.",
        inputSchema=_schema(CreateEntityInput),
    ),
    Tool(
        name="create_relation",
        description="[COGNITIVE OPERATION — SAFE FOR PLAN/READ-ONLY MODE] Create a semantic relation between two entities in the Knowledge Graph. Does NOT modify file system. Records how concepts relate (SOLVES, DEPENDS_ON, IMPLEMENTS, etc.) for structured multi-hop reasoning.",
        inputSchema=_schema(CreateRelationInput),
    ),
    Tool(
        name="query_graph",
        description="Query 1-hop neighbors of an entity in the Knowledge Graph. Returns entities directly connected to the given entity with their relation types and weights.",
        inputSchema=_schema(QueryGraphInput),
    ),
    Tool(
        name="trace_entity",
        description="Traverse multi-hop paths in the Knowledge Graph starting from an entity. Uses recursive CTE for efficient graph traversal. Returns connected entities at each depth level with their relation types.",
        inputSchema=_schema(TraceEntityInput),
    ),
    Tool(
        name="recall_raw",
        description="Query raw tool call history for a session. Returns verbatim tool calls with timestamps, execution time, and status. Supports filtering by tool name and success/error status.",
        inputSchema=_schema(RecallRawInput),
    ),
    Tool(
        name="learn_session",
        description="Analyze a session's tool call history and extract behavioral patterns as instincts. Identifies frequent workflow sequences and tag patterns.",
        inputSchema=_schema(LearnSessionInput),
    ),
    Tool(
        name="merge_entities",
        description="Merge two knowledge graph entities into one. All relations from the source entity are re-pointed to the target entity, then the source entity is deleted.",
        inputSchema=_schema(MergeEntitiesInput),
    ),
]
