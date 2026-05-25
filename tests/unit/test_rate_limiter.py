"""Tests for TokenBucketRateLimiter — per-session abuse protection."""

import pytest

from memorymesh.utils.rate_limiter import TokenBucketRateLimiter, get_global_limiter


@pytest.mark.asyncio
class TestTokenBucketRateLimiter:
    """Unit tests for the token bucket rate limiter."""

    async def test_initial_request_allowed(self):
        """First request from a key is always allowed."""
        limiter = TokenBucketRateLimiter(rate=10.0, window_seconds=60.0)
        assert await limiter.allow("session-1")

    async def test_burst_exhaustion(self):
        """After burst capacity is exhausted, requests are denied."""
        limiter = TokenBucketRateLimiter(rate=10.0, window_seconds=60.0, burst_ratio=0.5)
        key = "session-2"
        # Burst capacity = rate * burst_ratio = 5
        for _ in range(5):
            assert await limiter.allow(key)
        # 6th request should be denied
        assert not await limiter.allow(key)

    async def test_different_keys_independent(self):
        """Different session keys have independent buckets."""
        limiter = TokenBucketRateLimiter(rate=2.0, window_seconds=60.0, burst_ratio=0.5)
        for _ in range(1):  # consume 1 from key A
            assert await limiter.allow("session-a")
        assert await limiter.allow("session-b")  # Key B still has capacity

    async def test_reset_clears_bucket(self):
        """Reset clears the bucket for a key."""
        limiter = TokenBucketRateLimiter(rate=2.0, window_seconds=60.0, burst_ratio=0.5)
        key = "session-3"
        assert await limiter.allow(key)  # 1 used
        await limiter.reset(key)
        assert await limiter.allow(key)  # Full capacity again

    async def test_custom_cost(self):
        """Custom cost parameter reduces tokens accordingly."""
        limiter = TokenBucketRateLimiter(rate=10.0, window_seconds=60.0, burst_ratio=1.0)
        key = "heavy"
        assert await limiter.allow(key, cost=5.0)  # 10 - 5 = 5 remaining
        assert await limiter.allow(key, cost=5.0)  # 5 - 5 = 0
        assert not await limiter.allow(key)  # Exhausted

    async def test_cleanup_removes_stale_buckets(self):
        """Stale buckets are cleaned up after max_age_seconds."""
        limiter = TokenBucketRateLimiter(rate=10.0, window_seconds=60.0)
        await limiter.allow("stale-key")
        assert "stale-key" in limiter._buckets
        # Force the bucket to appear stale by setting last_refill far in the past
        limiter._buckets["stale-key"].last_refill = 0.0
        await limiter.cleanup(max_age_seconds=1.0)
        assert "stale-key" not in limiter._buckets

    async def test_global_limiter_singleton(self):
        """Global limiter is a singleton shared across invocations."""
        limiter1 = get_global_limiter()
        limiter2 = get_global_limiter()
        assert limiter1 is limiter2

    async def test_denied_response_format(self):
        """Denied request returns False consistently."""
        limiter = TokenBucketRateLimiter(rate=1.0, window_seconds=60.0, burst_ratio=0.0)
        key = "denied-session"
        assert not await limiter.allow(key)  # burst_ratio=0 means no initial tokens
