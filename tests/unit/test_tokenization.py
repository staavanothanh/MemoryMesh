import pytest

from memorymesh.utils.tokenization import estimate_tokens, count_tokens_exact, truncate_to_budget


class TestEstimateTokens:
    def test_short_text(self):
        assert estimate_tokens("hello world") == 2  # 11 // 4

    def test_empty_text(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_long_text(self):
        text = "x" * 1000
        assert estimate_tokens(text) == 250

    def test_ratio_approximate(self):
        text = "This is a test sentence with some words."
        est = estimate_tokens(text)
        assert est > 0


class TestTruncateToBudget:
    def test_under_budget(self):
        text = "short text"
        assert truncate_to_budget(text, 100) == text

    def test_over_budget(self):
        text = "x" * 4000
        result = truncate_to_budget(text, 10)  # ~10 tokens = ~40 chars
        assert len(result) < 4000
        assert "[truncated]" in result


@pytest.mark.asyncio
class TestCountTokensExact:
    async def test_basic_counting(self):
        count = await count_tokens_exact("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    async def test_empty_string(self):
        count = await count_tokens_exact("")
        assert count >= 0

    async def test_fallback_on_error(self):
        count = await count_tokens_exact("normal text")
        assert count > 0
