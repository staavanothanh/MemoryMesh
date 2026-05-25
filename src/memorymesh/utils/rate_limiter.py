"""Lightweight in-memory rate limiter for MCP tools using token bucket algorithm.

Prevents LLM infinite-loop abuse of expensive tools (recall, save_workspace_context, etc.)
by tracking per-session request counts with a sliding window.
"""

import time
import asyncio
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float = 0.0
    last_refill: float = 0.0


class TokenBucketRateLimiter:
    """Simple token-bucket rate limiter with per-key tracking.

    Default: 30 requests per 60 seconds per key, with burst capacity of 30.
    """

    def __init__(
        self,
        rate: float = 30.0,
        window_seconds: float = 60.0,
        burst_ratio: float = 1.0,
    ):
        self.rate = rate
        self.window = window_seconds
        self.capacity = rate * burst_ratio
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str, cost: float = 1.0) -> bool:
        """Check if the request is allowed. Returns True if permitted."""
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, last_refill=now)
                self._buckets[key] = bucket

            # Refill tokens based on elapsed time
            elapsed = now - bucket.last_refill
            refill = elapsed * (self.rate / self.window)
            bucket.tokens = min(self.capacity, bucket.tokens + refill)
            bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True
            return False

    async def reset(self, key: str):
        """Reset the bucket for a key (e.g., on session end)."""
        async with self._lock:
            self._buckets.pop(key, None)

    async def cleanup(self, max_age_seconds: float = 600.0):
        """Remove stale buckets older than max_age_seconds."""
        async with self._lock:
            now = time.monotonic()
            stale = [
                k for k, b in self._buckets.items()
                if now - b.last_refill > max_age_seconds
            ]
            for k in stale:
                self._buckets.pop(k, None)


# Per-session limiter instance — shared across all ToolHandlers
_global_limiter = TokenBucketRateLimiter(rate=30.0, window_seconds=60.0)


def get_global_limiter() -> TokenBucketRateLimiter:
    return _global_limiter
