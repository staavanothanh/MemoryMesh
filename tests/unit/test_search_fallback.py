import pytest
from unittest.mock import patch, AsyncMock

from memorymesh.schemas import SearchResult


class TestExtractQueryKeywords:
    def test_drops_vietnamese_question_words(self, memory_manager):
        result = memory_manager._extract_query_keywords("cuối buổi thảo luận gì")
        for w in result.split():
            assert w not in ("cuoi", "buoi", "thao"), f"{w} should be filtered"
        assert len(result.split()) <= 5

    def test_keeps_meaningful_words(self, memory_manager):
        result = memory_manager._extract_query_keywords("bug fix memory manager")
        meaningful = result.split()
        assert "bug" in meaningful
        assert "fix" in meaningful

    def test_top_5_only(self, memory_manager):
        result = memory_manager._extract_query_keywords(
            "one two three four five six seven eight"
        )
        assert len(result.split()) <= 5

    def test_skip_short_words(self, memory_manager):
        result = memory_manager._extract_query_keywords("a an at by bug")
        assert "bug" in result

    def test_empty_input(self, memory_manager):
        assert memory_manager._extract_query_keywords("") == ""

    def test_only_question_words(self, memory_manager):
        assert memory_manager._extract_query_keywords("what did we discuss") == ""

    def test_english_query_with_question_words(self, memory_manager):
        result = memory_manager._extract_query_keywords("what bug did we fix")
        words = result.split()
        assert "bug" in words
        assert "fix" in words
        assert "what" not in words


class TestSearchWithFallback:
    """Mock search_memory to test fallback logic in isolation."""

    @pytest.mark.asyncio
    async def test_tier1_semantic_returns_immediately(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            SearchResult(id="m1", content="test", score=0.95, tags=[], importance=3, timestamp=""),
        ])):
            results, tier, meta, _ = await memory_manager.search_with_fallback(
                query="test", top_k=5, user_id="test_user", min_score_threshold=0.3,
            )
            assert tier == "semantic"
            assert len(results) == 1
            assert results[0]["id"] == "m1"

    @pytest.mark.asyncio
    async def test_tier2_fts_as_fallback_when_semantic_empty(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords", return_value="keyword"):
                with patch.object(memory_manager.backend, "fts_search", new=AsyncMock(return_value=[
                    {"id": "m2", "content": "keyword match", "score": 1.0},
                ])):
                    with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                                      new=AsyncMock(return_value=[
                                          {"id": "m2", "metadata": {"tags": [], "level": "session"}},
                                      ])):
                        results, tier, meta, _ = await memory_manager.search_with_fallback(
                            query="keyword", top_k=5, user_id="test_user",
                        )
                        assert tier == "fts_keyword"
                        assert len(results) >= 1
                        assert results[0]["id"] == "m2"

    @pytest.mark.asyncio
    async def test_tier3_chronological_when_all_above_empty(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords", return_value="xyz"):
                with patch.object(memory_manager.backend, "fts_search", new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend, "list_recent", new=AsyncMock(return_value=[
                        {"id": "m3", "content": "recent", "score": 1.0},
                    ])):
                        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[
                                              {"id": "m3", "metadata": {"tags": [], "level": "session"}},
                                          ])):
                            results, tier, meta, _ = await memory_manager.search_with_fallback(
                                query="xyz", top_k=5, user_id="test_user",
                            )
                            assert tier == "chronological"
                            assert results[0]["id"] == "m3"

    @pytest.mark.asyncio
    async def test_all_tiers_empty_returns_empty(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            with patch.object(memory_manager, "_extract_query_keywords", return_value="xyz"):
                with patch.object(memory_manager.backend, "fts_search", new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend, "list_recent", new=AsyncMock(return_value=[])):
                        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[])):
                            results, tier, meta, _ = await memory_manager.search_with_fallback(
                                query="nothing", top_k=5, user_id="test_user",
                            )
                            assert tier == "empty"
                            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_tier_fts_uses_extracted_keywords(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[])):
            fts_mock = AsyncMock(return_value=[])
            with patch.object(memory_manager.backend, "fts_search", fts_mock):
                with patch.object(memory_manager, "_extract_query_keywords", return_value="bug fix"):
                    with patch.object(memory_manager.backend, "list_recent", new=AsyncMock(return_value=[])):
                        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[])):
                            await memory_manager.search_with_fallback(
                                query="what bug did we fix", top_k=5, user_id="test_user",
                            )
                            fts_mock.assert_called_once_with("bug fix", "test_user", limit=10)

    @pytest.mark.asyncio
    async def test_workspace_passed_to_tier1(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            SearchResult(id="m1", content="ws memory", score=0.9, tags=[], importance=3, timestamp=""),
        ])):
            results, tier, meta, _ = await memory_manager.search_with_fallback(
                query="test", top_k=5, user_id="test_user",
                workspace_path="/project/a",
            )
            assert tier == "semantic"
            memory_manager.search_memory.assert_called_with(
                query="test", top_k=5, user_id="test_user",
                workspace_path="/project/a", max_tokens=None,
            )

    @pytest.mark.asyncio
    async def test_max_tokens_passed_to_tier1(self, memory_manager):
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            SearchResult(id="m1", content="test", score=0.9, tags=[], importance=3, timestamp=""),
        ])):
            results, tier, meta, _ = await memory_manager.search_with_fallback(
                query="test", top_k=5, user_id="test_user", max_tokens=50,
            )
            assert tier == "semantic"
            memory_manager.search_memory.assert_called_with(
                query="test", top_k=5, user_id="test_user",
                workspace_path=None, max_tokens=50,
            )

    @pytest.mark.asyncio
    async def test_filtered_by_score_threshold(self, memory_manager):
        """Tier1 stays if at least one result passes threshold."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            SearchResult(id="high", content="high score", score=0.8, tags=[], importance=3, timestamp=""),
        ])):
            results, tier, meta, _ = await memory_manager.search_with_fallback(
                query="test", top_k=5, user_id="test_user", min_score_threshold=0.5,
            )
            assert tier == "semantic"
            assert len(results) == 1
            assert results[0]["id"] == "high"

    @pytest.mark.asyncio
    async def test_all_below_threshold_falls_through(self, memory_manager):
        """When all tier1 results are below threshold, fall through."""
        with patch.object(memory_manager, "search_memory", new=AsyncMock(return_value=[
            SearchResult(id="low", content="low score", score=0.2, tags=[], importance=3, timestamp=""),
        ])):
            with patch.object(memory_manager, "_extract_query_keywords", return_value="xyz"):
                with patch.object(memory_manager.backend, "fts_search", new=AsyncMock(return_value=[])):
                    with patch.object(memory_manager.backend, "list_recent", new=AsyncMock(return_value=[])):
                        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                                          new=AsyncMock(return_value=[])):
                            results, tier, meta, _ = await memory_manager.search_with_fallback(
                                query="test", top_k=5, user_id="test_user", min_score_threshold=0.5,
                            )
                            assert tier == "empty"


class TestEnrichFtsResults:
    @pytest.mark.asyncio
    async def test_attaches_metadata(self, memory_manager):
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(return_value=[
                              {"id": "m1", "metadata": {"tags": ["tag1"], "user_id": "test_user"}},
                          ])):
            fts_results = [{"id": "m1", "content": "test enrich", "score": 1.0}]
            enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
            assert len(enriched) == 1
            meta = enriched[0].get("metadata", {})
            assert meta.get("tags") == ["tag1"]

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, memory_manager):
        enriched = await memory_manager._enrich_fts_results([], "test_user")
        assert enriched == []

    @pytest.mark.asyncio
    async def test_handles_missing_enrichment(self, memory_manager):
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(return_value=[])):
            fts_results = [{"id": "nonexistent_id", "content": "test", "score": 1.0}]
            enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
            assert len(enriched) == 1
            assert enriched[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_enrichment_exception_graceful(self, memory_manager):
        with patch.object(memory_manager.backend, "get_with_embeddings_by_ids",
                          new=AsyncMock(side_effect=Exception("Enrichment failed"))):
            fts_results = [{"id": "m1", "content": "test", "score": 1.0}]
            enriched = await memory_manager._enrich_fts_results(fts_results, "test_user")
            assert len(enriched) == 1
            assert enriched[0]["metadata"] == {}
