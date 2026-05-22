import pytest
from unittest.mock import AsyncMock, MagicMock

from memorymesh.memory.fact_extractor import FactExtractor
from memorymesh.config import RouterConfig


@pytest.fixture
def fact_extractor(app_config):
    router = AsyncMock()
    router.call_llm = AsyncMock()
    router.call_llm_background = AsyncMock()
    return FactExtractor(app_config, router)


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_extract_facts_success(self, fact_extractor):
        fact_extractor.router.call_llm_background.return_value = '{"facts": [{"fact": "User likes Python", "confidence": "high", "tags": ["language"]}]}'
        facts = await fact_extractor.extract_facts("I like Python")
        assert len(facts) == 1
        assert facts[0]["fact"] == "User likes Python"
        assert facts[0]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_extract_facts_empty_input(self, fact_extractor):
        facts = await fact_extractor.extract_facts("")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_whitespace_only(self, fact_extractor):
        facts = await fact_extractor.extract_facts("   ")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_invalid_json(self, fact_extractor):
        fact_extractor.router.call_llm_background.return_value = "not json"
        facts = await fact_extractor.extract_facts("test")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_llm_error(self, fact_extractor):
        fact_extractor.router.call_llm_background.side_effect = RuntimeError("LLM down")
        facts = await fact_extractor.extract_facts("test")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_batch_success(self, fact_extractor):
        fact_extractor.router.call_llm_background.return_value = (
            '{"facts": [{"fact": "Fact A", "confidence": "high", "tags": ["a"]}, {"fact": "Fact B", "confidence": "medium", "tags": ["b"]}]}'
        )
        facts = await fact_extractor.extract_facts_batch(["Conversation 1", "Conversation 2"])
        assert len(facts) == 2
        assert facts[0]["fact"] == "Fact A"
        assert facts[1]["fact"] == "Fact B"

    @pytest.mark.asyncio
    async def test_extract_batch_empty(self, fact_extractor):
        facts = await fact_extractor.extract_facts_batch([])
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_batch_all_empty_strings(self, fact_extractor):
        facts = await fact_extractor.extract_facts_batch(["", ""])
        assert facts == []


class TestDeduplicateAndValidate:
    def test_deduplicate_identical(self, fact_extractor):
        raw = [
            {"fact": "User likes Python", "confidence": "high", "tags": []},
            {"fact": "User likes Python", "confidence": "medium", "tags": []},
        ]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert len(valid) == 1

    def test_deduplicate_case_insensitive(self, fact_extractor):
        raw = [
            {"fact": "User likes Python", "confidence": "high", "tags": []},
            {"fact": "user likes python", "confidence": "medium", "tags": []},
        ]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert len(valid) == 1

    def test_filter_empty_fact(self, fact_extractor):
        raw = [
            {"fact": "", "confidence": "high", "tags": []},
            {"fact": "Real fact", "confidence": "medium", "tags": []},
        ]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert len(valid) == 1
        assert valid[0]["fact"] == "Real fact"

    def test_filter_non_dict_item(self, fact_extractor):
        raw = ["not a dict", {"fact": "Real fact", "confidence": "high", "tags": []}]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert len(valid) == 1

    def test_confidence_fallback_to_medium(self, fact_extractor):
        raw = [{"fact": "Test", "confidence": 123, "tags": []}]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert valid[0]["confidence"] == "medium"

    def test_tags_fallback_to_empty(self, fact_extractor):
        raw = [{"fact": "Test", "confidence": "high", "tags": "not_a_list"}]
        valid = fact_extractor._deduplicate_and_validate(raw)
        assert valid[0]["tags"] == []

    def test_confidence_mapping(self, fact_extractor):
        assert fact_extractor._confidence_to_importance("high") == 4
        assert fact_extractor._confidence_to_importance("medium") == 3
        assert fact_extractor._confidence_to_importance("low") == 2
        assert fact_extractor._confidence_to_importance("unknown") == 3
