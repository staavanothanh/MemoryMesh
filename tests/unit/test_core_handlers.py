"""Tests for _core.py — _safe_error_response and _log_bg."""

import pytest
from memorymesh.mcp_server.handlers._core import _safe_error_response, _log_bg
from memorymesh.errors import MemoryMeshError


class TestSafeErrorResponse:
    def test_memory_mesh_error_exposes_message(self):
        """MemoryMeshError should expose its message in the response."""
        e = MemoryMeshError("Invalid input provided")
        result = _safe_error_response(e, "test_operation")
        assert result["status"] == "error"
        assert "Invalid input provided" in result["error"]

    def test_generic_exception_hides_details(self):
        """Non-MemoryMeshError should return generic error message."""
        e = ValueError("secret internal details")
        result = _safe_error_response(e, "test_operation")
        assert result["status"] == "error"
        assert result["error"] == "Internal server error"
        assert "secret" not in result["error"]

    def test_operation_label_in_error(self):
        """Operation label should be used in error logging."""
        e = MemoryMeshError("test error")
        result = _safe_error_response(e, "my_operation")
        assert result["status"] == "error"

    def test_no_operation_label(self):
        """Should work without operation label."""
        e = MemoryMeshError("test")
        result = _safe_error_response(e)
        assert result["status"] == "error"


class TestLogBg:
    def test_log_bg_executes_without_error(self):
        """_log_bg should not raise."""
        _log_bg("TestLabel", "test message", emoji="")
        _log_bg("TestLabel", "test message with emoji", emoji="🚀")
