"""Tests for Pydantic schema validation of MCP tool inputs."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from memorymesh.schemas import (
    TOOL_INPUT_MODELS,
    validate_tool_input,
    generate_tool_json_schema,
    RememberInput,
    RecallInput,
    ForgetInput,
    ArchiveMemoryInput,
    UnarchiveMemoryInput,
    ListMemoriesInput,
    SaveSystemPromptInput,
    CommitMilestoneInput,
    NewSessionInput,
    EndSessionInput,
    DeleteSessionInput,
    ResumeSessionInput,
    SearchResult,
    MemoryRecord,
    ToolOutput,
)


class TestRememberInput:
    """Validation for the remember MCP tool."""

    def test_valid_minimal_input(self):
        """Only required field 'content' should pass."""
        inp = RememberInput(content="Test memory")
        assert inp.content == "Test memory"
        assert inp.importance == 3
        assert inp.level == "user"
        assert inp.tags == []
        assert inp.workspace_path == ""

    def test_blank_content_raises(self):
        """Blank content should be rejected."""
        with pytest.raises(PydanticValidationError):
            RememberInput(content="   ")

    def test_content_too_short_raises(self):
        """Empty content should be rejected by min_length."""
        with pytest.raises(PydanticValidationError):
            RememberInput(content="")

    def test_invalid_importance_raises(self):
        """Importance outside 1-5 should be rejected."""
        with pytest.raises(PydanticValidationError):
            RememberInput(content="test", importance=10)

        with pytest.raises(PydanticValidationError):
            RememberInput(content="test", importance=0)

    def test_invalid_level_raises(self):
        """Level must be one of user/session/knowledge."""
        with pytest.raises(PydanticValidationError):
            RememberInput(content="test", level="invalid")

    def test_valid_full_input(self):
        """All optional fields with valid values."""
        inp = RememberInput(
            content="Important finding",
            tags=["bug", "critical"],
            importance=5,
            level="knowledge",
            workspace_path="/home/user/project",
            user_id="test_user",
        )
        assert inp.tags == ["bug", "critical"]
        assert inp.importance == 5
        assert inp.level == "knowledge"


class TestRecallInput:
    """Validation for the recall MCP tool."""

    def test_valid_minimal_input(self):
        inp = RecallInput(query="search term")
        assert inp.top_k == 5

    def test_top_k_out_of_range_raises(self):
        with pytest.raises(PydanticValidationError):
            RecallInput(query="test", top_k=0)
        with pytest.raises(PydanticValidationError):
            RecallInput(query="test", top_k=21)


class TestForgetInput:
    """Validation for the forget MCP tool."""

    def test_valid_input(self):
        inp = ForgetInput(memory_id="abc-123")
        assert inp.memory_id == "abc-123"

    def test_empty_memory_id_raises(self):
        with pytest.raises(PydanticValidationError):
            ForgetInput(memory_id="")


class TestArchiveMemoryInput:
    def test_valid_input(self):
        inp = ArchiveMemoryInput(memory_id="mem-456")
        assert inp.memory_id == "mem-456"


class TestUnarchiveMemoryInput:
    def test_valid_input(self):
        inp = UnarchiveMemoryInput(memory_id="mem-789")
        assert inp.memory_id == "mem-789"


class TestListMemoriesInput:
    def test_defaults(self):
        inp = ListMemoriesInput()
        assert inp.limit == 100
        assert inp.offset == 0

    def test_limit_out_of_range(self):
        with pytest.raises(PydanticValidationError):
            ListMemoriesInput(limit=2000)
        with pytest.raises(PydanticValidationError):
            ListMemoriesInput(limit=0)


class TestCommitMilestoneInput:
    def test_valid_input(self):
        inp = CommitMilestoneInput(
            summary="Fixed bug in login",
            tasks_done="patched auth.py",
            next_steps="Write tests",
        )
        assert inp.summary == "Fixed bug in login"

    def test_missing_summary_raises(self):
        with pytest.raises(PydanticValidationError):
            CommitMilestoneInput(summary="")


class TestNewSessionInput:
    def test_defaults(self):
        inp = NewSessionInput()
        assert inp.system_prompt == ""
        assert inp.workspace_path == ""


class TestEndSessionInput:
    def test_defaults(self):
        inp = EndSessionInput()
        assert inp.session_id == ""


class TestDeleteSessionInput:
    def test_defaults(self):
        inp = DeleteSessionInput()
        assert inp.session_id == ""


class TestResumeSessionInput:
    def test_valid_input(self):
        inp = ResumeSessionInput(session_id="sess-001")
        assert inp.top_k == 10

    def test_missing_session_id_raises(self):
        with pytest.raises(PydanticValidationError):
            ResumeSessionInput(session_id="")


class TestToolInputModelsMapping:
    """Verify all 18 tools have corresponding Pydantic models."""

    EXPECTED_TOOLS = {
        "remember", "recall", "forget", "archive_memory", "unarchive_memory",
        "list_memories", "ping", "save_system_prompt", "commit_milestone",
        "save_context_pair", "list_sessions", "get_session_context",
        "new_session", "end_session", "save_workspace_context",
        "delete_session", "preserve_session_memories", "resume_session",
    }

    def test_all_tools_have_models(self):
        assert set(TOOL_INPUT_MODELS.keys()) == self.EXPECTED_TOOLS

    def test_generate_schema_for_each_tool(self):
        for name in self.EXPECTED_TOOLS:
            schema = generate_tool_json_schema(name)
            assert "type" in schema
            assert schema["type"] == "object"


class TestValidateToolInput:
    """Integration of validate_tool_input helper."""

    def test_valid_tool_input(self):
        result = validate_tool_input("remember", {"content": "hello"})
        assert isinstance(result, RememberInput)
        assert result.content == "hello"

    def test_invalid_tool_input_raises(self):
        with pytest.raises(PydanticValidationError):
            validate_tool_input("remember", {"extra": "field"})

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            validate_tool_input("nonexistent", {})


class TestDomainModels:
    """Domain model validations (TypedDict, no Pydantic validation)."""

    def test_memory_record(self):
        rec: MemoryRecord = {
            "id": "mem-1",
            "user_id": "user-1",
            "content": "test",
            "importance": 4,
            "timestamp": "2025-01-01T00:00:00Z",
            "tags": [],
            "level": "user",
            "workspace_path": "",
        }
        assert rec["id"] == "mem-1"

    def test_search_result(self):
        sr: SearchResult = {
            "id": "mem-1",
            "content": "result",
            "score": 0.95,
            "importance": 3,
            "timestamp": "2025-01-01T00:00:00Z",
            "tags": [],
        }
        assert sr["score"] == 0.95

    def test_tool_output_success(self):
        out: ToolOutput = {"status": "success", "data": {"id": "abc"}}
        assert out["status"] == "success"
        assert out["data"] == {"id": "abc"}

    def test_tool_output_error(self):
        out: ToolOutput = {"status": "error", "error": "Something went wrong"}
        assert out["status"] == "error"
        assert out["error"] == "Something went wrong"
