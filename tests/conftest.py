import pytest
import time
from pathlib import Path
import tempfile
import shutil

from memorymesh.config import AppConfig, RouterConfig, SqliteVecConfig, ConsolidationConfig, SessionConfig
from memorymesh.router import RouterClient
from memorymesh.memory.sqlite_vec_backend import SqliteVecBackend
from memorymesh.memory.manager import MemoryManager


def _force_unlink(path: Path, retries: int = 5, delay: float = 0.1):
    for attempt in range(retries):
        try:
            path.unlink()
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
        except FileNotFoundError:
            return


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
def sqlite_vec_config():
    tmp_file = Path(tempfile.mktemp(suffix="_vec_test.db"))
    yield SqliteVecConfig(db_path=str(tmp_file))
    _force_unlink(tmp_file)


@pytest.fixture
def session_config():
    tmp_file = Path(tempfile.mktemp(suffix="_session_test.db"))
    yield SessionConfig(db_path=str(tmp_file), auto_create_session=False)
    _force_unlink(tmp_file)


@pytest.fixture
def app_config(router_config, sqlite_vec_config, session_config):
    return AppConfig(
        router=router_config,
        sqlite_vec=sqlite_vec_config,
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
def router_client(router_config):
    return RouterClient(router_config)


@pytest.fixture
async def backend(sqlite_vec_config):
    b = SqliteVecBackend(sqlite_vec_config.db_path)
    await b.initialize()
    yield b
    await b.close()


@pytest.fixture
async def memory_manager(app_config, backend, router_client):
    mgr = MemoryManager(app_config, backend, router_client)
    yield mgr
    await mgr.shutdown()
