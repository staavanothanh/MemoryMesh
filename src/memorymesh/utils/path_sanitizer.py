"""Workspace path sanitization to prevent path traversal attacks.

MCP tools receive workspace_path from LLM/client input. This module ensures
all paths stay within allowed boundaries before any file system access.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_BASE = os.getcwd()


def sanitize_workspace_path(path: str, base_dir: Optional[str] = None) -> str:
    """Validate and normalize a workspace path, ensuring it stays within base_dir.

    Args:
        path: Raw workspace_path from MCP tool arguments (may be empty/relative/absolute)
        base_dir: Allowed base directory (defaults to os.getcwd())

    Returns:
        Normalized absolute path guaranteed to be within base_dir

    Raises:
        ValueError: If path traversal outside base_dir is detected
    """
    if not path or not path.strip():
        return os.path.abspath(base_dir or DEFAULT_ALLOWED_BASE)

    allowed_base = os.path.abspath(base_dir or DEFAULT_ALLOWED_BASE)

    # Resolve relative paths against the allowed base first
    if not os.path.isabs(path):
        abs_path = os.path.realpath(os.path.join(allowed_base, path))
    else:
        abs_path = os.path.realpath(os.path.normpath(path))

    # Verify the resolved path stays within allowed boundaries
    if not is_path_within_base(abs_path, allowed_base):
        raise ValueError(
            f"Path traversal blocked: '{path}' resolves outside allowed directory"
        )

    return abs_path


def is_path_within_base(target: str, base: str) -> bool:
    """Check if target path is within base directory (prevents traversal).

    Uses os.path.commonpath on normalized, resolved paths for correctness
    across drive letters and symlinks on Windows.
    """
    target_norm = os.path.realpath(os.path.normpath(target))
    base_norm = os.path.realpath(os.path.normpath(base))

    try:
        common = os.path.commonpath([target_norm, base_norm])
        return os.path.normcase(common) == os.path.normcase(base_norm)
    except ValueError:
        # Different drives on Windows
        return False


def validate_directory_exists(path: str) -> bool:
    """Check that a sanitized workspace path points to an existing directory."""
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def safe_join_and_validate(base_dir: str, *subpaths: str) -> str:
    """Join subpaths under base_dir and validate the result stays within base.

    Args:
        base_dir: Sanitized base directory
        *subpaths: Additional path components to join

    Returns:
        Full normalized path within base_dir

    Raises:
        ValueError: If the joined path escapes base_dir
    """
    joined = os.path.join(base_dir, *subpaths)
    resolved = os.path.realpath(os.path.normpath(joined))
    if not is_path_within_base(resolved, base_dir):
        raise ValueError(
            f"Path traversal blocked when joining subpaths under '{base_dir}'"
        )
    return resolved
