import logging
from contextvars import ContextVar
from ...errors import MemoryMeshError

_session_var: ContextVar[str] = ContextVar('_session_var', default='')
_client_name_var: ContextVar[str] = ContextVar('_client_name_var', default='')

_ERROR_PRESERVE_KEYWORDS = frozenset({
    "fix", "bug", "error", "crash", "failure", "fail", "hotfix", "patch",
    "workaround", "root cause", "lỗi", "sửa", "debug",
})

# Bootstrap snapshot constants
_BOOTSTRAP_MAX_CHARS = 15000

_MAGENTA = "\033[1;35m"
_CYAN = "\033[1;36m"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"

logger = logging.getLogger(__name__)


def _log_bg(label: str, msg: str, emoji: str = ""):
    """ANSI-colored structured log for background operations."""
    logger.info("%s %s[%s]%s %s", emoji, _MAGENTA, label, _RESET, msg)


def _safe_error_response(e: Exception, operation: str = "") -> dict:
    """Convert an exception to a safe MCP error response without leaking internals.

    MemoryMeshError subclasses are considered safe to expose. All other exceptions
    are logged server-side and replaced with a generic error message.
    """
    if isinstance(e, MemoryMeshError):
        logger.error("%s failed: %s", operation or "Operation", e)
        return {"status": "error", "error": str(e)}
    logger.error("%s failed with unexpected error", operation or "Operation", exc_info=True)
    return {"status": "error", "error": "Internal server error"}
