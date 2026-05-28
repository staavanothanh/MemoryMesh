import json
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from memorymesh.mcp_server.handlers.base import ToolHandlers
from memorymesh.memory.session_store import SessionStore
from memorymesh.prompts import BOOTSTRAP_SNAPSHOT_PROMPT


@pytest_asyncio.fixture
async def tool_handlers(memory_manager, app_config):
    store = SessionStore(app_config.session.db_path)
    await store.initialize()
    handlers = ToolHandlers(memory_manager, store)
    yield handlers
    await store.close()


class TestCreateBootstrapSnapshot:
    @pytest.mark.asyncio
    async def test_format_includes_narrative_summary(self, tool_handlers):
        tool_handlers.manager.router.config.background_model_pool = ["test-model"]
        with patch.object(tool_handlers.manager.router, "_call",
                          new=AsyncMock(return_value=json.dumps({
                              "narrative_summary": "We fixed the search fallback system.",
                              "discussion_topic": "3-tier fallback optimization",
                              "architectural_decisions": "Keep HybridBackend, defer sqlite-vec",
                              "last_milestone": "Phase 6A test coverage done",
                              "next_steps": "Deploy and test",
                          }))):
            with patch.object(tool_handlers.session_store, "get_context_log",
                              new=AsyncMock(return_value=[
                                  {"role": "user", "content": "hello"},
                                  {"role": "assistant", "content": "hi"},
                                  {"role": "user", "content": "let's work"},
                                  {"role": "assistant", "content": "ok"},
                              ])):
                with patch.object(tool_handlers.manager, "add_memory",
                                  new=AsyncMock(return_value="mem_123")):
                    await tool_handlers._create_bootstrap_snapshot(
                        "test_session_id", "test_user",
                    )
                    add_call = tool_handlers.manager.add_memory.call_args
                    assert add_call is not None

                    kwargs = add_call[1]
                    text = kwargs.get("text", "")
                    tags = kwargs.get("tags", [])

                    assert "Narrative Summary" in text or "[Bootstrap]" in text
                    assert "narrative_summary" in text.lower() or "search fallback" in text
                    assert "bootstrap" in tags
                    assert "session_summary" in tags

    @pytest.mark.asyncio
    async def test_short_session_creates_fallback(self, tool_handlers):
        tool_handlers.manager.router.config.background_model_pool = ["test-model"]
        with patch.object(tool_handlers.manager.router, "_call",
                          new=AsyncMock(return_value=json.dumps({
                              "narrative_summary": "Session ended",
                              "discussion_topic": "",
                              "work_done": "",
                              "architectural_decisions": "",
                              "last_milestone": "Session ended",
                              "next_steps": "",
                          }))):
            with patch.object(tool_handlers.session_store, "get_context_log",
                              new=AsyncMock(return_value=[
                                  {"role": "user", "content": "hi"},
                              ])):
                with patch.object(tool_handlers.manager, "add_memory",
                                  new=AsyncMock(return_value="mem_123")):
                    await tool_handlers._create_bootstrap_snapshot(
                        "test_session_id", "test_user",
                    )
                    add_call = tool_handlers.manager.add_memory.call_args
                    assert add_call is not None
                    kwargs = add_call[1]
                    assert "bootstrap" in kwargs.get("tags", [])

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback_dict(self, tool_handlers):
        tool_handlers.manager.router.config.background_model_pool = ["test-model"]
        with patch.object(tool_handlers.manager.router, "_call",
                          new=AsyncMock(side_effect=Exception("LLM down"))):
            with patch.object(tool_handlers.session_store, "get_context_log",
                              new=AsyncMock(return_value=[
                                  {"role": "user", "content": "hello"},
                                  {"role": "assistant", "content": "hi"},
                              ])):
                with patch.object(tool_handlers.manager, "add_memory",
                                  new=AsyncMock(return_value="mem_123")):
                    await tool_handlers._create_bootstrap_snapshot(
                        "test_session_id", "test_user",
                    )
                    add_call = tool_handlers.manager.add_memory.call_args
                    assert add_call is not None
                    kwargs = add_call[1]
                    text = kwargs.get("text", "")
                    assert "ended" in text.lower()


class TestGetBootstrapScaffold:
    @pytest.mark.asyncio
    async def test_prefers_bootstrap_tagged_result(self, tool_handlers):
        """When results include a bootstrap-tagged memory, prefer it over higher-scored ones."""
        with patch.object(tool_handlers.manager, "search_with_fallback",
                          new=AsyncMock(return_value=(
                              [
                                  {"id": "m1", "content": "non-bootstrap result",
                                   "tags": ["narrative_thread"], "score": 0.9},
                                  {"id": "m2", "content": "bootstrap snapshot result",
                                   "tags": ["bootstrap", "workspace_state"], "score": 0.7},
                              ],
                              "semantic",
                              {},
                              None,
                          ))):
            scaffold = await tool_handlers._get_bootstrap_scaffold(
                "test_user", "/project",
            )
            assert scaffold is not None
            assert "bootstrap snapshot result" in scaffold

    @pytest.mark.asyncio
    async def test_prefers_session_summary_tag(self, tool_handlers):
        with patch.object(tool_handlers.manager, "search_with_fallback",
                          new=AsyncMock(return_value=(
                              [
                                  {"id": "m1", "content": "some narrative",
                                   "tags": ["conversation"], "score": 0.8},
                                  {"id": "m2", "content": "session summary data",
                                   "tags": ["session_summary"], "score": 0.6},
                              ],
                              "semantic",
                              {},
                              None,
                          ))):
            scaffold = await tool_handlers._get_bootstrap_scaffold(
                "test_user", "/project",
            )
            assert scaffold is not None
            assert "session summary data" in scaffold

    @pytest.mark.asyncio
    async def test_fallback_to_first_result_when_no_tag_match(self, tool_handlers):
        with patch.object(tool_handlers.manager, "search_with_fallback",
                          new=AsyncMock(return_value=(
                              [
                                  {"id": "m1", "content": "first result content",
                                   "tags": ["random"], "score": 0.9},
                                  {"id": "m2", "content": "second result",
                                   "tags": ["random2"], "score": 0.8},
                              ],
                              "semantic",
                              {},
                              None,
                          ))):
            scaffold = await tool_handlers._get_bootstrap_scaffold(
                "test_user", "/project",
            )
            assert scaffold is not None
            assert "first result content" in scaffold

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self, tool_handlers):
        with patch.object(tool_handlers.manager, "search_with_fallback",
                          new=AsyncMock(return_value=([], "semantic", {}, None))):
            scaffold = await tool_handlers._get_bootstrap_scaffold(
                "test_user", "/project",
            )
            assert scaffold is None

    @pytest.mark.asyncio
    async def test_includes_cognitive_protocol(self, tool_handlers):
        with patch.object(tool_handlers.manager, "search_with_fallback",
                          new=AsyncMock(return_value=(
                              [{"id": "m1", "content": "test content",
                                "tags": ["bootstrap"], "score": 0.9}],
                              "semantic",
                              {},
                              None,
                          ))):
            scaffold = await tool_handlers._get_bootstrap_scaffold(
                "test_user", "/project",
            )
            assert "CONTEXT RESTORATION" in scaffold
            assert "INITIALIZATION COMPLETE" in scaffold
            assert "PAST SESSION CONTEXT" in scaffold
            assert "snapshot includes" in scaffold
            assert "Do NOT" not in scaffold


class TestBootstrapPrompt:
    def test_prompt_includes_narrative_summary(self):
        assert "narrative_summary" in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_includes_work_done(self):
        assert "work_done" in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_excludes_project_identity(self):
        assert "project_identity" not in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_excludes_open_impediments(self):
        assert "open_impediments" not in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_includes_next_steps(self):
        assert "next_steps" in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_requires_json_output(self):
        assert "valid JSON" in BOOTSTRAP_SNAPSHOT_PROMPT

    def test_prompt_forbids_markdown_and_filler(self):
        assert "markdown" in BOOTSTRAP_SNAPSHOT_PROMPT
