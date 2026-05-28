"""Test uncovered paths in config.py: validate() methods, from_env()."""

import pytest
import os

from memorymesh.config import (
    RouterConfig,
    SqliteVecConfig,
    SessionConfig,
    ConsolidationConfig,
    InstinctConfig,
    EccIntegrationConfig,
    EmbeddingConfig,
    AppConfig,
)


class TestRouterConfigValidate:
    def test_valid_url(self):
        config = RouterConfig(url="http://localhost:9999/v1")
        config.validate()  # Should not raise

    def test_invalid_url_raises(self):
        config = RouterConfig(url="")
        with pytest.raises(AssertionError, match="Invalid router URL"):
            config.validate()


class TestSqliteVecConfigValidate:
    def test_valid_config(self, tmp_path):
        db_path = os.path.join(tmp_path, "test.db")
        config = SqliteVecConfig(db_path=db_path, max_concurrent_writes=5)
        config.validate()  # Should not raise — creates directory

    def test_invalid_concurrent_writes_raises(self):
        config = SqliteVecConfig(db_path="./db/memory.db", max_concurrent_writes=0)
        with pytest.raises(AssertionError, match="max_concurrent_writes must be 1"):
            config.validate()

    def test_too_many_concurrent_writes_raises(self):
        config = SqliteVecConfig(db_path="./db/memory.db", max_concurrent_writes=11)
        with pytest.raises(AssertionError):
            config.validate()


class TestSessionConfigValidate:
    def test_valid_config(self, tmp_path):
        db_path = os.path.join(tmp_path, "sessions.db")
        config = SessionConfig(db_path=db_path)
        config.validate()  # Should not raise

    def test_creates_directory(self, tmp_path):
        db_path = os.path.join(tmp_path, "subdir", "sessions.db")
        config = SessionConfig(db_path=db_path)
        config.validate()
        assert os.path.exists(os.path.dirname(db_path))


class TestConsolidationConfigValidate:
    def test_valid_config(self):
        config = ConsolidationConfig()
        config.validate()  # Should not raise

    def test_invalid_threshold_raises(self):
        config = ConsolidationConfig(similarity_threshold=0.0)
        with pytest.raises(AssertionError, match="similarity_threshold"):
            config.validate()

    def test_invalid_threshold_above_1_raises(self):
        config = ConsolidationConfig(similarity_threshold=1.5)
        with pytest.raises(AssertionError, match="similarity_threshold"):
            config.validate()

    def test_invalid_min_cluster_size_raises(self):
        config = ConsolidationConfig(min_cluster_size=1)
        with pytest.raises(AssertionError, match="min_cluster_size"):
            config.validate()

    def test_invalid_min_importance_raises(self):
        config = ConsolidationConfig(min_importance_to_keep=0)
        with pytest.raises(AssertionError, match="min_importance_to_keep"):
            config.validate()

    def test_invalid_ephemeral_ttl_raises(self):
        config = ConsolidationConfig(ephemeral_memory_ttl_days=-1)
        with pytest.raises(AssertionError, match="ephemeral_memory_ttl_days"):
            config.validate()


class TestInstinctConfigValidate:
    def test_valid_config(self, tmp_path):
        db_path = os.path.join(tmp_path, "instincts.db")
        config = InstinctConfig(db_path=db_path)
        config.validate()  # Should not raise

    def test_creates_directory(self, tmp_path):
        db_path = os.path.join(tmp_path, "nested", "instincts.db")
        config = InstinctConfig(db_path=db_path)
        config.validate()
        assert os.path.exists(os.path.dirname(db_path))

    def test_invalid_v2_max_instincts_raises(self):
        config = InstinctConfig(v2_max_instincts=0)
        with pytest.raises(AssertionError, match="v2_max_instincts"):
            config.validate()

    def test_invalid_max_pattern_length_raises(self):
        config = InstinctConfig(max_pattern_length=5)
        with pytest.raises(AssertionError, match="max_pattern_length"):
            config.validate()

    def test_invalid_confidence_floor_raises(self):
        config = InstinctConfig(confidence_floor=-0.1)
        with pytest.raises(AssertionError, match="confidence_floor"):
            config.validate()

    def test_invalid_dedup_similarity_raises(self):
        config = InstinctConfig(dedup_similarity_threshold=0.4)
        with pytest.raises(AssertionError, match="dedup_similarity_threshold"):
            config.validate()


class TestEccIntegrationConfigValidate:
    def test_valid_config(self):
        config = EccIntegrationConfig()
        config.validate()  # Should not raise (empty)


class TestEmbeddingConfigValidate:
    def test_valid_local(self):
        config = EmbeddingConfig(mode="local")
        config.validate()  # Should not raise

    def test_remote_without_url_raises(self):
        config = EmbeddingConfig(mode="remote", remote_api_url="")
        with pytest.raises(ValueError, match="REMOTE_EMBEDDING_API_URL"):
            config.validate()

    def test_remote_with_url_succeeds(self):
        config = EmbeddingConfig(mode="remote", remote_api_url="http://localhost:9999")
        config.validate()  # Should not raise


class TestAppConfigValidate:
    def test_validate_delegates_to_sub_configs(self, tmp_path):
        config = AppConfig(
            router=RouterConfig(url="http://localhost:9999/v1"),
            sqlite_vec=SqliteVecConfig(db_path=os.path.join(tmp_path, "mem.db")),
            session=SessionConfig(db_path=os.path.join(tmp_path, "sess.db")),
            consolidation=ConsolidationConfig(),
            instinct=InstinctConfig(db_path=os.path.join(tmp_path, "inst.db")),
        )
        config.validate()  # Should not raise

    def test_from_env_loads_defaults(self):
        """from_env loads env vars with defaults."""
        config = AppConfig.from_env()
        assert config.default_user_id is not None
        assert config.mcp_transport in ("stdio", "sse")
        assert config.router.timeout_s > 0
