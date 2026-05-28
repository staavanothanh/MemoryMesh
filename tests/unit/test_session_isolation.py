"""Tests for session isolation between MCP connections.

Verifies that each ToolHandlers instance has its own session ID,
concurrent access to _session_map is safe, and clearing sessions works.
"""

import asyncio
import pytest

from memorymesh.mcp_server.handlers.base import ToolHandlers


class TestSessionPerConnection:
    """Each ToolHandlers instance must have independent session state."""

    @pytest.mark.asyncio
    async def test_session_per_connection(self, memory_manager):
        """Create two ToolHandlers instances with different connection keys
        and session IDs — each returns its own session ID."""
        # Create a minimal SessionStore for ToolHandlers
        from memorymesh.memory.session_store import SessionStore
        import tempfile
        from pathlib import Path

        tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
        try:
            store = SessionStore(str(tmp_file))
            await store.initialize()

            handler1 = ToolHandlers(memory_manager, store)
            handler2 = ToolHandlers(memory_manager, store)

            # Set different connection keys
            handler1._connection_key = "conn_1"
            handler2._connection_key = "conn_2"

            # Set different session IDs
            await handler1.set_session("session-a")
            await handler2.set_session("session-b")

            # Verify each returns its own
            sid1 = await handler1.get_current_session_id()
            sid2 = await handler2.get_current_session_id()

            assert sid1 == "session-a", f"Handler1 should have 'session-a', got '{sid1}'"
            assert sid2 == "session-b", f"Handler2 should have 'session-b', got '{sid2}'"
            assert sid1 != sid2, "Sessions should be different"

            await store.close()
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_default_connection_is_empty(self, memory_manager):
        """No _connection_key set returns empty string."""
        from memorymesh.memory.session_store import SessionStore
        import tempfile
        from pathlib import Path

        tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
        try:
            store = SessionStore(str(tmp_file))
            await store.initialize()

            handler = ToolHandlers(memory_manager, store)
            # Don't set _connection_key — should default to ""
            sid = await handler.get_current_session_id()
            assert sid == "", "Default session ID should be empty string"

            await store.close()
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)


class TestSessionMapConcurrentAccess:
    """Multiple concurrent set_session / get_current_session_id calls."""

    @pytest.mark.asyncio
    async def test_session_map_concurrent_access(self, memory_manager):
        """Multiple concurrent set_session/get_current_session_id calls
        with different connection keys — no data races."""
        from memorymesh.memory.session_store import SessionStore
        import tempfile
        from pathlib import Path

        tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
        try:
            store = SessionStore(str(tmp_file))
            await store.initialize()

            handler = ToolHandlers(memory_manager, store)
            n_connections = 20

            async def set_and_get(conn_idx: int):
                key = f"conn_{conn_idx}"
                session_id = f"session_{conn_idx}"
                handler._connection_key = key
                await handler.set_session(session_id)
                result = await handler.get_current_session_id()
                return (key, result)

            results = await asyncio.gather(*[
                set_and_get(i) for i in range(n_connections)
            ])

            for key, result in results:
                expected_suffix = key.split("_")[1]
                assert result == f"session_{expected_suffix}", (
                    f"Connection '{key}' expected session_{expected_suffix}, got '{result}'"
                )

            await store.close()
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)


class TestSessionClear:
    """Setting empty session_id cleans up the map entry."""

    @pytest.mark.asyncio
    async def test_session_clear(self, memory_manager):
        """Setting empty session_id removes the entry from _session_map."""
        from memorymesh.memory.session_store import SessionStore
        import tempfile
        from pathlib import Path

        tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
        try:
            store = SessionStore(str(tmp_file))
            await store.initialize()

            handler = ToolHandlers(memory_manager, store)
            handler._connection_key = "conn_test"

            # Set a session
            await handler.set_session("test-session")
            sid = await handler.get_current_session_id()
            assert sid == "test-session"

            # Clear by setting empty string
            await handler.set_session("")
            sid = await handler.get_current_session_id()
            assert sid == "", "After clearing, session should be empty"

            # Map should not have the key anymore
            assert "conn_test" not in handler._session_map

            await store.close()
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_session_clear_nonexistent_key(self, memory_manager):
        """Clearing a session for a key that doesn't exist should be a no-op."""
        from memorymesh.memory.session_store import SessionStore
        import tempfile
        from pathlib import Path

        tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
        try:
            store = SessionStore(str(tmp_file))
            await store.initialize()

            handler = ToolHandlers(memory_manager, store)
            # Set connection key but don't set any session
            handler._connection_key = "nonexistent_key"

            # Clearing should work without error
            await handler.set_session("")
            sid = await handler.get_current_session_id()
            assert sid == ""

            await store.close()
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
