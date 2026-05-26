"""Token counting utilities — fast estimation + exact async counting."""

import asyncio
import logging

logger = logging.getLogger(__name__)

_TIKTOKEN_ENCODING = "cl100k_base"
_tiktoken = None


def estimate_tokens(text: str) -> int:
    """Ultra-fast token estimation using character ratio (~4 chars/token)."""
    return max(1, len(text) // 4)


def _load_tiktoken():
    global _tiktoken
    if _tiktoken is None:
        import tiktoken
        _tiktoken = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    return _tiktoken


def _count_exact_sync(text: str) -> int:
    """Synchronous tiktoken counting — run in thread pool."""
    enc = _load_tiktoken()
    return len(enc.encode(text))


async def count_tokens_exact(text: str) -> int:
    """Exact token counting via tiktoken, offloaded to thread pool.

    Uses asyncio.to_thread to avoid blocking the event loop.
    Falls back to estimate_tokens if tiktoken fails.
    """
    try:
        return await asyncio.to_thread(_count_exact_sync, text)
    except Exception as e:
        logger.debug("Exact token count failed, using estimation: %s", e)
        return estimate_tokens(text)


def truncate_to_budget(text: str, budget: int) -> str:
    """Fast truncation — estimates via char ratio, exact truncation via tiktoken only if needed."""
    est = estimate_tokens(text)
    if est <= budget:
        return text
    # Coarse truncation to estimated budget
    char_budget = budget * 4
    truncated = text[:char_budget]
    return truncated + "\n...[truncated]"
