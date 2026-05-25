"""Tests for path sanitization utility — prevents path traversal attacks."""

import os
import tempfile
import pytest
from pathlib import Path

from memorymesh.utils.path_sanitizer import (
    sanitize_workspace_path,
    is_path_within_base,
    validate_directory_exists,
    safe_join_and_validate,
)


class TestSanitizeWorkspacePath:
    """Validate that workspace paths stay within allowed boundaries."""

    def test_empty_path_returns_current_dir(self):
        """Empty path falls back to current working directory."""
        result = sanitize_workspace_path("")
        assert os.path.isabs(result)
        assert result == os.path.abspath(os.getcwd())

    def test_none_path_returns_current_dir(self):
        """None/empty strip falls back to cwd."""
        result = sanitize_workspace_path("   ")
        assert os.path.isabs(result)

    def test_relative_path_under_base(self, tmp_path):
        """Relative path resolves within base directory."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        result = sanitize_workspace_path("sub", base_dir=str(tmp_path))
        assert result == str(subdir.resolve())

    def test_absolute_path_within_base(self, tmp_path):
        """Absolute path within base is accepted."""
        subdir = tmp_path / "valid"
        subdir.mkdir()
        result = sanitize_workspace_path(str(subdir), base_dir=str(tmp_path))
        assert os.path.normcase(result) == os.path.normcase(str(subdir.resolve()))

    def test_traversal_outside_base_raises(self, tmp_path):
        """Path traversal outside allowed base raises ValueError."""
        outside = tempfile.gettempdir()
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_workspace_path(outside, base_dir=str(tmp_path))

    def test_parent_dir_traversal_raises(self, tmp_path):
        """Using ../ to escape base raises ValueError."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_workspace_path("../../../etc", base_dir=str(tmp_path))

    def test_custom_base_dir(self, tmp_path):
        """Custom base directory is respected."""
        base = tmp_path / "workspace"
        base.mkdir()
        result = sanitize_workspace_path(".", base_dir=str(base))
        assert os.path.normcase(result) == os.path.normcase(str(base.resolve()))


class TestIsPathWithinBase:
    """Unit tests for the core containment check."""

    def test_same_path(self, tmp_path):
        assert is_path_within_base(str(tmp_path), str(tmp_path))

    def test_child_path(self, tmp_path):
        child = tmp_path / "child"
        child.mkdir()
        assert is_path_within_base(str(child), str(tmp_path))

    def test_parent_path_rejected(self, tmp_path):
        parent = tmp_path.parent
        assert not is_path_within_base(str(parent), str(tmp_path))

    def test_sibling_path_rejected(self, tmp_path):
        sibling = Path(tempfile.mkdtemp())
        try:
            assert not is_path_within_base(str(sibling), str(tmp_path))
        finally:
            sibling.rmdir()


class TestValidateDirectoryExists:
    """Validation that a path points to an existing directory."""

    def test_existing_dir(self, tmp_path):
        assert validate_directory_exists(str(tmp_path))

    def test_nonexistent_dir(self, tmp_path):
        assert not validate_directory_exists(str(tmp_path / "nope"))

    def test_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert not validate_directory_exists(str(f))


class TestSafeJoinAndValidate:
    """Safe path joining with containment validation."""

    def test_valid_join(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        result = safe_join_and_validate(str(tmp_path), "sub")
        assert result == str(sub.resolve())

    def test_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Path traversal"):
            safe_join_and_validate(str(tmp_path), "..", "..", "etc")
