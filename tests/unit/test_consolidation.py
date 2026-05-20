import pytest
from unittest.mock import AsyncMock, patch

from memorymesh.memory.consolidation import ConsolidationEngine


SAMPLE_EMBEDDING = [0.1] * 384


@pytest.fixture
def consolidation(app_config):
    return ConsolidationEngine(app_config, None, None)


class TestCosineSim:
    def test_identical(self, consolidation):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert consolidation._cosine_sim(a, b) == pytest.approx(1.0)

    def test_orthogonal(self, consolidation):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert consolidation._cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite(self, consolidation):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert consolidation._cosine_sim(a, b) == pytest.approx(-1.0)

    def test_partial(self, consolidation):
        a = [1.0, 0.0]
        b = [0.5, 0.5]
        sim = consolidation._cosine_sim(a, b)
        assert 0.5 < sim < 1.0

    def test_zero_vector(self, consolidation):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert consolidation._cosine_sim(a, b) == pytest.approx(0.0)


class TestFindClusters:
    def test_no_clusters_below_threshold(self, consolidation):
        memories = [
            {"embedding": [1.0, 0.0], "metadata": {}},
            {"embedding": [0.0, 1.0], "metadata": {}},
        ]
        consolidation.threshold = 0.9
        clusters = consolidation._find_clusters(memories)
        # Each memory in its own cluster since sim=0 < 0.9
        assert len(clusters) == 2
        assert all(len(c) == 1 for c in clusters)

    def test_identical_embeddings_cluster(self, consolidation):
        memories = [
            {"embedding": [1.0, 0.0], "metadata": {}},
            {"embedding": [1.0, 0.0], "metadata": {}},
            {"embedding": [0.0, 1.0], "metadata": {}},
        ]
        consolidation.threshold = 0.9
        clusters = consolidation._find_clusters(memories)
        # First two identical should cluster, third separate
        cluster_sizes = sorted(len(c) for c in clusters)
        assert cluster_sizes == [1, 2]


class TestRunForUser:
    @pytest.mark.asyncio
    async def test_disabled_returns_zero(self, app_config):
        app_config.consolidation.enabled = False
        engine = ConsolidationEngine(app_config, None, None)
        result = await engine.run_for_user("test_user")
        assert result == 0

    @pytest.mark.asyncio
    async def test_run_merges_clusters(self, app_config):
        app_config.consolidation.enabled = True
        engine = ConsolidationEngine(app_config, None, None)

        mock_backend = AsyncMock()
        mock_backend.get_with_embeddings.return_value = [
            {"id": "m1", "content": "Alice likes cats", "embedding": [1.0, 0.0, 0.0], "metadata": {"importance": 3}},
            {"id": "m2", "content": "Alice loves feline pets", "embedding": [1.0, 0.0, 0.0], "metadata": {"importance": 4}},
            {"id": "m3", "content": "Bob likes dogs", "embedding": [0.0, 1.0, 0.0], "metadata": {"importance": 3}},
        ]
        mock_backend.add.return_value = "new_id"
        mock_backend.update_metadata.return_value = True

        mock_router = AsyncMock()
        mock_router.call_llm.return_value = '{"content": "Alice likes cats and other feline pets", "tags": ["pets", "cats"], "importance": 4}'

        with patch("memorymesh.memory.consolidation.get_embedding", return_value=[0.5, 0.5, 0.0]):
            engine.backend = mock_backend
            engine.router = mock_router
            engine.threshold = 0.9

            result = await engine.run_for_user("test_user")
            assert result == 1
            assert mock_backend.add.call_count == 1
            assert mock_backend.update_metadata.call_count >= 2
