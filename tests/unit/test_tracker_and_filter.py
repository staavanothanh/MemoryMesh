import time
import pytest
import pytest_asyncio

from memorymesh.mcp_server.handlers import SemanticFilter, ConversationTracker


class TestSemanticFilter:
    def test_short_content_is_noise(self):
        assert not SemanticFilter.is_valuable("hi")
        assert not SemanticFilter.is_valuable("ok")
        assert not SemanticFilter.is_valuable("")

    def test_known_noise_patterns(self):
        for phrase in ("hello", "thanks", "okay", "got it",
                       "cảm ơn", "được", "vâng", "ừ"):
            assert not SemanticFilter.is_valuable(phrase)

    def test_meaningful_content_passes(self):
        assert SemanticFilter.is_valuable("We need to refactor the search fallback system")
        assert SemanticFilter.is_valuable("The bug is in the router retry logic")

    def test_boundary_20_chars(self):
        assert not SemanticFilter.is_valuable("a" * 19)
        assert SemanticFilter.is_valuable("a" * 20 + " meaningful")


class TestConversationTracker:
    def test_initial_state(self):
        t = ConversationTracker("session_123")
        assert t.session_id == "session_123"
        assert t._uncommitted_actions == 0
        assert t._has_unsaved_context is False
        assert t._hostage_data is None
        assert not t.should_flush()

    def test_record_tool_call_updates_activity_and_context(self):
        t = ConversationTracker("session_123")
        old_time = t._last_activity
        t.record_tool_call("remember", {"content": "test fact"})
        assert t._last_activity >= old_time
        assert "[remember:test fact]" in t._tool_footprints
        assert t._has_unsaved_context is True

    def test_record_tool_call_captures_footprint(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("recall", {"query": "what did we do last"})
        assert "[recall:what did we do last]" in t._tool_footprints

    def test_record_tool_call_increments_actions_for_mutate(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "ls"})
        t.record_tool_call("write_file", {"content": "test"})
        assert t._uncommitted_actions == 2  # not in READ_ONLY_TOOLS

    def test_record_tool_call_readonly_does_not_increment_actions(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("ping", {})
        t.record_tool_call("list_memories", {})
        # READ_ONLY tools still get footprint (for context) but don't increment actions
        assert len(t._tool_footprints) == 2
        assert t._uncommitted_actions == 0  # no increment for read-only

    def test_record_tool_call_skips_commit_milestone(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("commit_milestone", {"summary": "test"})
        assert t._uncommitted_actions == 0  # commit_milestone doesn't count

    def test_record_tool_call_increments_for_remember(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("forget", {"memory_id": "abc"})
        assert t._uncommitted_actions == 1

    def test_on_milestone_commit_resets(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "echo hello"})
        t.record_tool_call("forget", {"memory_id": "abc"})
        assert t._uncommitted_actions == 2

        t.on_milestone_commit()
        assert len(t._tool_footprints) == 0
        assert t._uncommitted_actions == 0
        assert t._has_unsaved_context is False

    def test_engage_and_resolve_choke_point(self):
        t = ConversationTracker("session_123")
        t._uncommitted_actions = 5  # reach threshold

        msg = t.engage_choke_point({"data": "important results"})
        assert "CHOKE-POINT" in msg
        assert "5 uncommitted actions" in msg
        assert t._hostage_data == {"data": "important results"}

        released = t.resolve_milestone()
        assert released == {"data": "important results"}
        assert t._hostage_data is None
        assert t._uncommitted_actions == 0

    def test_should_flush_false_when_active(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "test"})
        assert not t.should_flush(idle_threshold=30.0)

    def test_should_flush_true_when_idle_and_actions_pending(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "test"})
        t._last_activity = 0
        assert t.should_flush(idle_threshold=0.1)

    def test_should_flush_false_when_idle_but_no_actions(self):
        t = ConversationTracker("session_123")
        t._last_activity = 0
        assert not t.should_flush(idle_threshold=0.1)  # _uncommitted_actions == 0

    def test_flush_returns_snapshot_text(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("recall", {"query": "project context"})
        t.record_tool_call("bash", {"command": "ls"})

        snapshot = t.flush()
        assert "[AUTO-SNAPSHOT]" in snapshot
        assert "[recall:project context]" in snapshot

    def test_flush_resets_footprints(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "ls"})
        t.flush()
        assert len(t._tool_footprints) == 0

    def test_teardown_flush_returns_none_when_no_unsaved_context(self):
        t = ConversationTracker("session_123")
        result = t.teardown_flush()
        assert result is None

    def test_teardown_flush_returns_snapshot_when_unsaved(self):
        t = ConversationTracker("session_123")
        t.record_tool_call("bash", {"command": "ls"})
        result = t.teardown_flush()
        assert result is not None
        assert "[SESSION-FINAL-SNAPSHOT]" in result
        assert t._has_unsaved_context is False  # reset after flush
