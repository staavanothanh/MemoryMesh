import pytest
import asyncio
import sqlite3
import json
import zlib
from memorymesh.memory.async_batch_logger import AsyncBatchLogger

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE raw_log (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            tool_name TEXT,
            input_payload BLOB,
            output_payload BLOB,
            is_compressed INTEGER,
            execution_time_ms REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
    return str(db_path)

@pytest.mark.asyncio
async def test_async_batch_logger_basic(temp_db):
    logger = AsyncBatchLogger(db_path=temp_db, batch_size=2, flush_interval=0.1)
    logger.start()
    
    await logger.log_event("session1", "tool1", {"in": 1}, {"out": 1}, 10.0, "success")
    await logger.log_event("session1", "tool2", {"in": 2}, {"out": 2}, 20.0, "error")
    
    # Wait for flush
    await asyncio.sleep(0.3)
    
    # Check DB
    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT session_id, tool_name, input_payload, output_payload, status FROM raw_log").fetchall()
    conn.close()
    
    assert len(rows) == 2
    assert rows[0][0] == "session1"
    assert rows[0][1] == "tool1"
    assert rows[0][4] == "success"
    
    # Check compression
    in_payload = zlib.decompress(rows[0][2]).decode('utf-8')
    assert json.loads(in_payload) == {"in": 1}
    
    await logger.stop()

@pytest.mark.asyncio
async def test_async_batch_logger_timeout_flush(temp_db):
    logger = AsyncBatchLogger(db_path=temp_db, batch_size=10, flush_interval=0.1)
    logger.start()
    
    await logger.log_event("session1", "tool1", {"in": 1}, {"out": 1}, 10.0, "success")
    
    # Not reached batch size, but should flush after interval
    await asyncio.sleep(0.3)
    
    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT * FROM raw_log").fetchall()
    conn.close()
    
    assert len(rows) == 1
    
    await logger.stop()

@pytest.mark.asyncio
async def test_async_batch_logger_stop_flushes(temp_db):
    logger = AsyncBatchLogger(db_path=temp_db, batch_size=10, flush_interval=5.0)
    logger.start()
    
    await logger.log_event("session1", "tool1", {"in": 1}, {"out": 1}, 10.0, "success")
    await asyncio.sleep(0.1) # yield to let the worker pull from the queue
    
    await logger.stop()
    
    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT * FROM raw_log").fetchall()
    conn.close()
    
    assert len(rows) == 1
    
@pytest.mark.asyncio
async def test_async_batch_logger_stop_no_start():
    # Should not error if stopped without starting
    logger = AsyncBatchLogger(db_path="dummy", batch_size=10, flush_interval=5.0)
    await logger.stop()

def test_compress_sync_fallback():
    batch = [{
        'session_id': 's1',
        'tool_name': 't1',
        'input': 'invalid \ud800 string',
        'output': 'ok',
        'execution_time_ms': 1.0,
        'status': 'ok'
    }]
    compressed = AsyncBatchLogger._compress_sync(batch)
    assert len(compressed) == 1
    assert zlib.decompress(compressed[0][2]).decode('utf-8') == '{}'

def test_write_sync_error(caplog):
    try:
        AsyncBatchLogger._write_sync("/invalid/path/db.sqlite", [])
    except sqlite3.OperationalError:
        pass
