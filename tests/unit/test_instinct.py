import pytest
from unittest.mock import AsyncMock

from memorymesh.memory.instinct import InstinctEngine


def _make_memory(tags, content="x y"):
    """Create a mock memory dict. Content uses short words to avoid keyword instincts."""
    return {
        "id": "mem1",
        "user_id": "Shinn",
        "content": content,
        "metadata": {"tags": tags},
        "created_at": "2026-05-01T00:00:00",
    }


@pytest.fixture
def instinct_engine(app_config):
    backend = AsyncMock()
    store = AsyncMock()
    store.count_active = AsyncMock(return_value=0)
    store.get_active_instincts = AsyncMock(return_value=[])
    store.add_instinct = AsyncMock(return_value="instinct-1")
    return InstinctEngine(app_config, backend, store)


class TestLearnFromRecent:
    @pytest.mark.asyncio
    async def test_not_enough_memories(self, instinct_engine):
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 4
        await instinct_engine.learn_from_recent("Shinn")
        instinct_engine.store.add_instinct.assert_not_called()

    @pytest.mark.asyncio
    async def test_enough_memories_learns_tags(self, instinct_engine):
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 10
        await instinct_engine.learn_from_recent("Shinn")
        tag_calls = [
            c for c in instinct_engine.store.add_instinct.call_args_list
            if c[1]["condition"]["type"] == "tag_frequency"
        ]
        assert len(tag_calls) == 1
        assert tag_calls[0][1]["condition"]["tag"] == "python"
        assert tag_calls[0][1]["confidence"] >= 0.3

    @pytest.mark.asyncio
    async def test_tag_co_occurrence_detected(self, instinct_engine):
        memories = [_make_memory(["python", "flask"])] * 10 + [_make_memory(["python"])] * 5
        instinct_engine.backend.list_all.return_value = memories
        await instinct_engine.learn_from_recent("Shinn")
        tag_calls = [
            c for c in instinct_engine.store.add_instinct.call_args_list
            if c[1]["condition"]["type"] == "tag_frequency"
        ]
        assert len(tag_calls) >= 2

    @pytest.mark.asyncio
    async def test_instinct_limit_skips(self, instinct_engine):
        instinct_engine.store.count_active.return_value = 50
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 10
        await instinct_engine.learn_from_recent("Shinn")
        instinct_engine.store.add_instinct.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_duplicate_tag_instincts(self, instinct_engine):
        existing = [
            {
                "id": "e1",
                "condition": {"type": "tag_frequency", "tag": "python", "min_count": 3},
                "confidence": 0.5,
            }
        ]
        instinct_engine.store.get_active_instincts.return_value = existing
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 10
        await instinct_engine.learn_from_recent("Shinn")
        tag_calls = [
            c for c in instinct_engine.store.add_instinct.call_args_list
            if c[1]["condition"]["type"] == "tag_frequency"
        ]
        assert len(tag_calls) == 0

    @pytest.mark.asyncio
    async def test_confidence_scales_with_count(self, instinct_engine):
        instinct_engine.backend.list_all.return_value = [_make_memory(["rust"])] * 5
        await instinct_engine.learn_from_recent("Shinn")
        calls_5 = instinct_engine.store.add_instinct.call_args_list
        conf_5 = [c for c in calls_5 if c[1]["condition"]["type"] == "tag_frequency"][0][1]["confidence"]

        instinct_engine.store.add_instinct.reset_mock()
        instinct_engine.backend.list_all.return_value = [_make_memory(["rust"])] * 10
        await instinct_engine.learn_from_recent("Shinn")
        calls_10 = instinct_engine.store.add_instinct.call_args_list
        conf_10 = [c for c in calls_10 if c[1]["condition"]["type"] == "tag_frequency"][0][1]["confidence"]

        assert conf_10 > conf_5

    @pytest.mark.asyncio
    async def test_confidence_capped_at_09(self, instinct_engine):
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 100
        await instinct_engine.learn_from_recent("Shinn")
        tag_calls = [
            c for c in instinct_engine.store.add_instinct.call_args_list
            if c[1]["condition"]["type"] == "tag_frequency"
        ]
        assert len(tag_calls) >= 1
        for c in tag_calls:
            assert c[1]["confidence"] <= 0.9


class TestKeywordLearning:
    def test_extract_keywords(self, instinct_engine):
        words = instinct_engine._extract_keywords("I love Python programming")
        assert "python" in words
        assert "programming" in words
        assert "love" in words

    def test_extract_keywords_skip_short(self, instinct_engine):
        words = instinct_engine._extract_keywords("a an at by x y")
        assert all(len(w) >= 3 for w in words)

    def test_extract_keywords_skip_stopwords(self, instinct_engine):
        words = instinct_engine._extract_keywords("the and for are but you")
        assert all(w not in ["the", "and", "for"] for w in words)


class TestReinforce:
    @pytest.mark.asyncio
    async def test_reinforce_confidence_unchanged_by_learn(self, instinct_engine):
        existing = [
            {"id": "e1", "condition": {"type": "tag_frequency", "tag": "python", "min_count": 3}, "confidence": 0.5}
        ]
        instinct_engine.store.get_active_instincts.return_value = existing
        instinct_engine.backend.list_all.return_value = [_make_memory(["python"])] * 10
        await instinct_engine.learn_from_recent("Shinn")
        instinct_engine.store.add_instinct.assert_not_called()

    @pytest.mark.asyncio
    async def test_reinforce_instinct_success(self, instinct_engine):
        instinct_engine.store.update_confidence = AsyncMock()
        await instinct_engine.reinforce_instinct("inst-1", success=True)
        instinct_engine.store.update_confidence.assert_called_once_with("inst-1", 0.0, increment_trigger=True)

    @pytest.mark.asyncio
    async def test_reinforce_instinct_failure(self, instinct_engine):
        instinct_engine.store.update_confidence = AsyncMock()
        await instinct_engine.reinforce_instinct("inst-1", success=False)
        instinct_engine.store.update_confidence.assert_called_once_with("inst-1", -0.05, increment_trigger=False)
