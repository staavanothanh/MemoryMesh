import pytest
from pathlib import Path
import tempfile
import shutil

from memorymesh.config import AppConfig, RouterConfig, ChromaConfig, FTSConfig, ConsolidationConfig, SessionConfig
from memorymesh.router import RouterClient
from memorymesh.memory.chroma_impl import ChromaMemoryBackend
from memorymesh.memory.hybrid_backend import HybridBackend
from memorymesh.memory.manager import MemoryManager


@pytest.fixture
def router_config():
    return RouterConfig(
        url="http://127.0.0.1:20128/v1",
        default_model="test-model",
        fallback_model="test-fallback",
        timeout_s=5,
        max_retries=1,
    )


@pytest.fixture
def chroma_config():
    tmp_dir = Path(tempfile.mkdtemp(suffix="_chroma_test"))
    yield ChromaConfig(db_path=str(tmp_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def fts_config():
    tmp_file = Path(tempfile.mktemp(suffix="_fts_test.db"))
    yield FTSConfig(db_path=str(tmp_file))
    if tmp_file.exists():
        tmp_file.unlink()


@pytest.fixture
def session_config():
    tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
    yield SessionConfig(db_path=str(tmp_file), auto_create_session=False)
    if tmp_file.exists():
        tmp_file.unlink()


@pytest.fixture
def app_config(router_config, chroma_config, fts_config, session_config):
    return AppConfig(
        router=router_config,
        chroma=chroma_config,
        fts=fts_config,
        session=session_config,
        consolidation=ConsolidationConfig(
            similarity_threshold=0.85,
            min_cluster_size=2,
            interval_seconds=3600,
            batch_size=50,
            enabled=False,
            session_memory_ttl_days=7,
        ),
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        default_user_id="test_user",
        log_level="DEBUG",
    )


@pytest.fixture
def chroma_backend(chroma_config):
    return ChromaMemoryBackend(chroma_config.db_path)


@pytest.fixture
def router_client(router_config):
    return RouterClient(router_config)


@pytest.fixture
async def hybrid_backend(app_config):
    backend = HybridBackend(app_config)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def memory_manager(app_config, hybrid_backend, router_client):
    mgr = MemoryManager(app_config, hybrid_backend, router_client)
    yield mgr
    await mgr.shutdown()
