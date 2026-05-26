import asyncio
import zlib
import json
import time
import logging
import sqlite3
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class AsyncBatchLogger:
    """Async batch writer for raw tool call logs with zlib compression.
    
    Uses an unbounded queue for zero-latency log submission, batches by 
    count or time, and offloads compression + DB writes to thread pool.
    """
    
    def __init__(self, db_path: str, batch_size: int = 50, flush_interval: float = 1.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
    
    def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
    
    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
    
    async def log_event(self, session_id: str, tool_name: str, input_dict: dict, output_dict: dict, exec_time_ms: float, status: str):
        await self._queue.put({
            'session_id': session_id,
            'tool_name': tool_name,
            'input': json.dumps(input_dict),
            'output': json.dumps(output_dict),
            'execution_time_ms': exec_time_ms,
            'status': status,
        })
    
    async def _worker(self):
        batch = []
        last_flush = time.monotonic()
        while True:
            try:
                timeout = max(0.0, self.flush_interval - (time.monotonic() - last_flush))
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                batch.append(item)
                if len(batch) >= self.batch_size:
                    await self._process_and_write(batch)
                    batch = []
                    last_flush = time.monotonic()
            except asyncio.TimeoutError:
                if batch:
                    await self._process_and_write(batch)
                    batch = []
                last_flush = time.monotonic()
            except asyncio.CancelledError:
                if batch:
                    await self._process_and_write(batch)
                raise
            except Exception as e:
                logger.error("AsyncBatchLogger worker error: %s", e)
    
    @staticmethod
    def _compress_sync(batch: list) -> list:
        compressed = []
        for event in batch:
            try:
                c_input = zlib.compress(event['input'].encode('utf-8'))
            except Exception:
                c_input = zlib.compress(b'{}')
            try:
                c_output = zlib.compress(event['output'].encode('utf-8'))
            except Exception:
                c_output = zlib.compress(b'{}')
            compressed.append((
                event['session_id'],
                event['tool_name'],
                sqlite3.Binary(c_input),
                sqlite3.Binary(c_output),
                1,
                event['execution_time_ms'],
                event['status'],
            ))
        return compressed
    
    @staticmethod
    def _write_sync(db_path: str, compressed_batch: list):
        conn = sqlite3.connect(db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executemany(
                """INSERT INTO raw_log (session_id, tool_name, input_payload, output_payload, is_compressed, execution_time_ms, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                compressed_batch,
            )
            conn.commit()
        except Exception as e:
            logger.error("AsyncBatchLogger write failed: %s", e)
        finally:
            conn.close()
    
    async def _process_and_write(self, batch: list):
        compressed_batch = await asyncio.to_thread(self._compress_sync, batch)
        await asyncio.to_thread(self._write_sync, self.db_path, compressed_batch)
