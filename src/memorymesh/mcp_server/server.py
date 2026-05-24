import asyncio
import json
import os
import signal
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn

from ..config import AppConfig
from ..router import RouterClient
from ..hooks import hooks as global_hooks
from ..memory.sqlite_vec_backend import SqliteVecBackend
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..logging_ import setup_logging
from ..prompts import COMBINED_AGENT_INSTRUCTION
from ..embedder import prewarm_embedder
from .tools import TOOLS
from .handlers import ToolHandlers, SemanticFilter

logger = logging.getLogger(__name__)

PID_FILE = "memorymesh.pid"


def _ensure_single_instance(db_dir: str):
    """Kill orphaned MCP server process and write PID file."""
    pid_file = os.path.join(db_dir, PID_FILE)
    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, signal.SIGTERM)
                for _ in range(50):
                    try:
                        os.kill(old_pid, 0)
                        time.sleep(0.1)
                    except (OSError, ProcessLookupError):
                        break
            except (OSError, ProcessLookupError):
                pass
        except (ValueError, OSError):
            pass
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid_file(db_dir: str):
    pid_file = os.path.join(db_dir, PID_FILE)
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
    except OSError:
        pass


class MemoryMeshServer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.backend = SqliteVecBackend(config.sqlite_vec.db_path)
        self.router = RouterClient(config.router)
        self.manager = MemoryManager(config, self.backend, self.router, hooks=global_hooks)
        self.session_store = SessionStore(config.session.db_path)
        self.handlers = ToolHandlers(self.manager, self.session_store)
        self.mcp_server = Server("memorymesh")
        self._shutdown_event = asyncio.Event()
        self._background_tasks: set[asyncio.Task] = set()
        self._idle_flush_task: Optional[asyncio.Task] = None
        self._register_tools()

    def _init_options(self):
        opts = self.mcp_server.create_initialization_options()
        return opts.model_copy(update={"instructions": COMBINED_AGENT_INSTRUCTION})

    def _register_tools(self):
        @self.mcp_server.list_tools()
        async def list_tools() -> list[Tool]:
            return TOOLS

        @self.mcp_server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            handler_map = {
                "remember": self.handlers.handle_remember,
                "recall": self.handlers.handle_recall,
                "forget": self.handlers.handle_forget,
                "archive_memory": self.handlers.handle_archive_memory,
                "unarchive_memory": self.handlers.handle_unarchive_memory,
                "list_memories": self.handlers.handle_list_memories,
                "ping": self.handlers.handle_ping,
                "save_system_prompt": self.handlers.handle_save_system_prompt,
                "commit_milestone": self.handlers.handle_commit_milestone,
                "save_context_pair": self.handlers.handle_save_context_pair,
                "list_sessions": self.handlers.handle_list_sessions,
                "get_session_context": self.handlers.handle_get_session_context,
                "new_session": self.handlers.handle_new_session,
                "end_session": self.handlers.handle_end_session,
                "delete_session": self.handlers.handle_delete_session,
                "preserve_session_memories": self.handlers.handle_preserve_session_memories,
                "save_workspace_context": self.handlers.handle_save_workspace_context,
                "resume_session": self.handlers.handle_resume_session,
            }
            handler = handler_map.get(name)
            if not handler:
                raise ValueError(f"Unknown tool: {name}")

            # Detect client from MCP request context (per-connection, via contextvars)
            client_name = ""
            try:
                ctx = self.mcp_server.request_context
                client_caps = ctx.session.client_params
                if client_caps and client_caps.client_info:
                    client_name = client_caps.client_info.name or ""
            except (LookupError, AttributeError):
                pass

            # Auto-init session if none active (skip for new_session itself to avoid recursion)
            if name != "new_session":
                await self.handlers.ensure_session(arguments)

            # Layer 2/3 tracking (must run before handler)
            await self.handlers._note_tool_call(name, arguments)

            result = await handler(arguments)

            # LAYER 2: Auto-save every tool call (except read-only to avoid overwrite loop)
            read_only_tools = ("ping", "list_sessions", "list_memories", "get_session_context", "recall")
            if name not in read_only_tools:
                asyncio.create_task(
                    self._safely_auto_save_tool(name, arguments, result)
                )

            # Build response content array
            contents = [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False)
            )]

            return contents

    def _create_tracked_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _safely_auto_save_tool(self, tool_name: str, args: dict, result: dict):
        try:
            await self.handlers.save_auto_tool_context(tool_name, args, result)
        except Exception as e:
            logger.warning("Layer 2 auto-save failed: %s", e)

    async def _initialize_fast(self):
        """Fast init: open DBs, warm embedder, and create a fresh session."""
        await self.backend.initialize()
        await self.session_store.initialize()
        if self.config.instinct.enabled:
            await self.manager.instinct_store.initialize()
        # Pre-warm embedding model synchronously so first recall is fast
        await prewarm_embedder(self.config.embedding_model)
        if self.config.session.auto_create_session:
            system_prompt = COMBINED_AGENT_INSTRUCTION
            session_id = await self.session_store.create_session(
                self.config.default_user_id,
                system_prompt=system_prompt,
            )
            logger.info("Auto-created fresh session: %s", session_id)
            await self.handlers.set_session(session_id)
        self.handlers.start_write_worker()
        self._idle_flush_task = self._create_tracked_task(self.handlers._idle_flush_loop())

    async def _initialize_slow(self):
        """Slow background init: model-dependent tasks (auto-scan, auto-recall)."""
        try:
            if self.config.session.auto_scan_codebase:
                await self.handlers._auto_scan_codebase(user_id=self.config.default_user_id)
        except Exception as e:
            logger.warning("Slow initialization failed: %s", e)

    async def _cleanup(self):
        logger.info("Shutting down...")

        # TEARDOWN HOOK: flush unsaved context before shutdown
        try:
            await self.handlers._trigger_teardown_snapshot()
        except Exception as e:
            logger.warning("Teardown snapshot error: %s", e)

        # Stop background tasks
        await self.handlers.stop_write_worker()
        try:
            self._idle_flush_task.cancel()
        except Exception:
            pass

        # Flush any pending trackers before shutdown (teardown already captured the main one)
        try:
            async with self.handlers._tracker_lock:
                session_ids = list(self.handlers._trackers.keys())
            for sid in session_ids:
                async with self.handlers._tracker_lock:
                    tracker = self.handlers._trackers.get(sid)
                if tracker:
                    snapshot = tracker.teardown_flush()
                    if snapshot and SemanticFilter.is_valuable(snapshot):
                        logger.info("Shutdown flush session %s", sid[:8])
                        try:
                            await self.manager.add_memory(
                                text=snapshot,
                                tags=["auto_save", "shutdown", f"session:{sid}"],
                                importance=2,
                                level="session",
                                user_id=self.config.default_user_id,
                            )
                        except Exception as e:
                            logger.warning("Shutdown flush write failed: %s", e)
        except Exception as e:
            logger.warning("Shutdown flush error: %s", e)

        # Drain memory write queue
        try:
            while not self.handlers._write_queue.empty():
                task = self.handlers._write_queue.get_nowait()
                await self.manager.add_memory(**task)
        except Exception:
            pass

        try:
            current_id = await self.handlers.get_current_session_id()
            if current_id:
                logger.info("Finalizing active session before shutdown: %s", current_id)
                uid = self.config.default_user_id
                await asyncio.wait_for(
                    self.handlers._finalize_session(current_id, uid),
                    timeout=30.0
                )
        except asyncio.TimeoutError:
            logger.warning("Shutdown finalization timed out (30s)")
        except Exception as e:
            logger.warning("Shutdown finalization failed: %s", e)

        try:
            await self.backend.close()
        except Exception as e:
            logger.warning("Backend close error: %s", e)
        try:
            await self.session_store.close()
        except Exception as e:
            logger.warning("Session store close error: %s", e)
        try:
            await self.manager.instinct_store.close()
        except Exception as e:
            logger.warning("Instinct store close error: %s", e)
        try:
            await self.router.close()
        except Exception as e:
            logger.warning("Router close error: %s", e)
        try:
            await self.manager.shutdown()
        except Exception as e:
            logger.warning("Manager shutdown error: %s", e)
        logger.info("Shutdown complete")

    async def run_stdio(self):
        await self._initialize_fast()
        async with stdio_server() as (read_stream, write_stream):
            self._create_tracked_task(self._initialize_slow())
            await self.mcp_server.run(
                read_stream,
                write_stream,
                self._init_options(),
            )

    async def run_sse(self):
        await self._initialize_fast()
        await self._initialize_slow()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.mcp_server.run(
                    streams[0],
                    streams[1],
                    self._init_options(),
                )

        async def health(request):
            from starlette.responses import JSONResponse
            return JSONResponse({"status": "ok", "service": "memorymesh"})

        app = Starlette(
            routes=[
                Route("/health", endpoint=health),
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
            on_shutdown=[self._cleanup],
        )
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.mcp_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run(self):
        async def _run():
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(
                        sig,
                        lambda: asyncio.create_task(self._shutdown()),
                    )
                except NotImplementedError:
                    pass
            try:
                if self.config.mcp_transport == "sse":
                    await self.run_sse()
                else:
                    await self.run_stdio()
            except Exception as e:
                logger.error("Server error: %s", e)
            finally:
                await self._cleanup()

        asyncio.run(_run())

    async def _shutdown(self):
        logger.warning("Received shutdown signal, finalizing current session...")
        try:
            await self.handlers._trigger_teardown_snapshot()
        except Exception:
            pass
        try:
            current_id = await self.handlers.get_current_session_id()
            if current_id:
                uid = self.config.default_user_id
                await asyncio.wait_for(
                    self.handlers._finalize_session(current_id, uid),
                    timeout=30.0,
                )
        except asyncio.TimeoutError:
            logger.warning("Shutdown finalization timed out (30s), force closing...")
        except Exception as e:
            logger.warning("Shutdown finalization failed: %s", e)
        self._shutdown_event.set()


def main():
    config = AppConfig.from_env()
    config.validate()
    setup_logging(config.log_level)
    db_dir = os.path.dirname(config.session.db_path)
    _ensure_single_instance(db_dir)
    try:
        server = MemoryMeshServer(config)
        server.run()
    finally:
        _remove_pid_file(db_dir)