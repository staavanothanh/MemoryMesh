"""Tests for Pydantic extra="forbid" on all MCP tool input models.

Ensures user_id is always preserved through validation and no silent field
drops occur when unknown fields are passed to tool input models.
"""

import pytest
from pydantic import BaseModel, ValidationError as PydanticValidationError

from memorymesh.schemas import (
    TOOL_INPUT_MODELS,
    BaseToolInput,
    validate_tool_input,
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


class TestBaseToolInput:
    """BaseToolInput is the root of all tool input models with extra='forbid'."""

    def test_base_tool_input_defaults(self):
        """Assert BaseToolInput() has user_id=''."""
        inp = BaseToolInput()
        assert inp.user_id == "", "Default user_id should be empty string"

    def test_base_tool_input_with_user_id(self):
        inp = BaseToolInput(user_id="Alice")
        assert inp.user_id == "Alice"


class TestAllInputModelsExtendBase:
    """All 24 tool input models must extend BaseToolInput."""

    def test_all_input_models_extend_base(self):
        """Iterate TOOL_INPUT_MODELS registry, assert each is subclass of BaseToolInput."""
        # Expected number of models (including merge_entities)
        assert len(TOOL_INPUT_MODELS) >= 24, "Expected at least 24 registered models"
        for name, model_cls in TOOL_INPUT_MODELS.items():
            assert issubclass(model_cls, BaseToolInput), (
                f"Model '{name}' ({model_cls.__name__}) does not extend BaseToolInput"
            )


class TestUserIdPreserved:
    """user_id must survive round-trip through validate_tool_input."""

    @pytest.mark.parametrize("tool_name,extra_args", [
        ("remember", {"content": "test"}),
        ("recall", {"query": "test query"}),
        ("forget", {"memory_id": "mem-1"}),
        ("archive_memory", {"memory_id": "mem-1"}),
        ("unarchive_memory", {"memory_id": "mem-1"}),
        ("list_memories", {}),
        ("ping", {}),
        ("save_system_prompt", {"system_prompt": "hello"}),
        ("commit_milestone", {"summary": "test"}),
        ("save_context_pair", {"user_message": "hello"}),
        ("list_sessions", {}),
        ("get_session_context", {"session_id": "sess-1"}),
        ("new_session", {}),
        ("end_session", {}),
        ("save_workspace_context", {}),
        ("delete_session", {}),
        ("preserve_session_memories", {}),
        ("resume_session", {"session_id": "sess-1"}),
        ("create_entity", {"name": "Test"}),
        ("create_relation", {"source": "A", "target": "B", "relation_type": "USES"}),
        ("query_graph", {"entity_name": "Test"}),
        ("trace_entity", {"entity_name": "Test"}),
        ("recall_raw", {}),
        ("learn_session", {}),
        ("merge_entities", {"source": "A", "target": "B"}),
    ])
    def test_user_id_preserved_through_validation(self, tool_name, extra_args):
        """Call validate_tool_input, dump result, verify user_id == 'Alice'."""
        args = {**extra_args, "user_id": "Alice"}
        result = validate_tool_input(tool_name, args)
        assert result.user_id == "Alice", (
            f"user_id='Alice' not preserved for tool '{tool_name}'; got '{result.user_id}'"
        )

    def test_user_id_default_empty_when_omitted(self):
        """When user_id is not provided, it should default to empty string."""
        result = validate_tool_input("remember", {"content": "test"})
        assert result.user_id == "", "Default user_id should be empty string"


class TestExtraFieldsRejected:
    """extra='forbid' must reject any field not defined in the model."""

    @pytest.mark.parametrize("tool_name,valid_args", [
        ("remember", {"content": "test"}),
        ("recall", {"query": "test"}),
        ("forget", {"memory_id": "mem-1"}),
        ("archive_memory", {"memory_id": "mem-1"}),
        ("unarchive_memory", {"memory_id": "mem-1"}),
        ("list_memories", {}),
        ("ping", {}),
        ("save_system_prompt", {"system_prompt": "hello"}),
        ("commit_milestone", {"summary": "test"}),
        ("save_context_pair", {"user_message": "test"}),
        ("list_sessions", {}),
        ("get_session_context", {"session_id": "sess-1"}),
        ("new_session", {}),
        ("end_session", {}),
        ("save_workspace_context", {}),
        ("delete_session", {}),
        ("preserve_session_memories", {}),
        ("resume_session", {"session_id": "sess-1"}),
        ("create_entity", {"name": "Test"}),
        ("create_relation", {"source": "A", "target": "B", "relation_type": "USES"}),
        ("query_graph", {"entity_name": "Test"}),
        ("trace_entity", {"entity_name": "Test"}),
        ("recall_raw", {}),
        ("learn_session", {}),
        ("merge_entities", {"source": "A", "target": "B"}),
    ])
    def test_extra_fields_rejected(self, tool_name, valid_args):
        """Call with unknown_field='x' — assert ValidationError."""
        args = {**valid_args, "unknown_field": "x"}
        with pytest.raises(PydanticValidationError):
            validate_tool_input(tool_name, args)

    def test_extra_field_on_base_tool_input(self):
        """BaseToolInput itself also has extra='forbid'."""
        with pytest.raises(PydanticValidationError):
            BaseToolInput(unknown_field="x")


class TestSpecificModelValidation:
    """Additional validation checks on specific input models."""

    def test_remember_input_with_tags_and_level(self):
        inp = RememberInput(content="test", tags=["a", "b"], level="session")
        assert inp.tags == ["a", "b"]
        assert inp.level == "session"

    def test_recall_input_with_cursor(self):
        inp = RecallInput(query="test", cursor='{"last_score": 0.5, "last_id": "x", "page": 2}')
        assert inp.cursor is not None

    def test_recall_input_invalid_cursor_raises(self):
        with pytest.raises(PydanticValidationError):
            RecallInput(query="test", cursor="not-json")

    def test_forget_input_preserves_user_id(self):
        inp = ForgetInput(memory_id="mem-1", user_id="Bob")
        assert inp.user_id == "Bob"

    def test_ping_input_empty(self):
        inp = PingInput()
        assert inp.user_id == ""
