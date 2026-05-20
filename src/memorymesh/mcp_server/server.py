import asyncio
import os
import signal
import logging
import time
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
from ..memory.hybrid_backend import HybridBackend
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..logging_ import setup_logging
from ..prompts import RECALL_INSTRUCTION
from .tools import TOOLS
from .handlers import ToolHandlers

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
        self.backend = HybridBackend(config)
        self.router = RouterClient(config.router)
        self.manager = MemoryManager(config, self.backend, self.router, hooks=global_hooks)
        self.session_store = SessionStore(config.session.db_path)
        self.handlers = ToolHandlers(self.manager, self.session_store)
        self.mcp_server = Server("memorymesh")
        self._shutdown_event = asyncio.Event()
        self._register_tools()

    def _init_options(self):
        opts = self.mcp_server.create_initialization_options()
        return opts.model_copy(update={"instructions": RECALL_INSTRUCTION})

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
                "list_memories": self.handlers.handle_list_memories,
                "ping": self.handlers.handle_ping,
                "save_system_prompt": self.handlers.handle_save_system_prompt,
                "save_context_pair": self.handlers.handle_save_context_pair,
                "list_sessions": self.handlers.handle_list_sessions,
                "get_session_context": self.handlers.handle_get_session_context,
                "new_session": self.handlers.handle_new_session,
                "end_session": self.handlers.handle_end_session,
                "save_workspace_context": self.handlers.handle_save_workspace_context,
                "resume_session": self.handlers.handle_resume_session,
            }
            handler = handler_map.get(name)
            if not handler:
                raise ValueError(f"Unknown tool: {name}")
            result = await handler(arguments)
            return [TextContent(type="text", text=str(result))]

    async def _initialize_fast(self):
        """Fast init: open DBs and restore/resume session."""
        await self.backend.initialize()
        await self.session_store.initialize()
        if self.config.instinct.enabled:
            await self.manager.instinct_store.initialize()
        if self.config.session.auto_create_session:
            active = await self.session_store.get_active_session(self.config.default_user_id)
            if active:
                session_id = active["session_id"]
                logger.info("Resuming active session: %s", session_id)
            else:
                session_id = await self.session_store.create_session(self.config.default_user_id)
                logger.info("Auto-created session: %s", session_id)
            await self.handlers.set_session(session_id)

    async def _initialize_slow(self):
        """Slow background init: model-dependent tasks (auto-scan, auto-recall)."""
        try:
            if self.config.session.auto_scan_codebase:
                await self.handlers._auto_scan_codebase(user_id=self.config.default_user_id)
        except Exception as e:
            logger.warning("Slow initialization failed: %s", e)

    async def _cleanup(self):
        logger.info("Shutting down...")
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
        logger.info("Shutdown complete")

    async def run_stdio(self):
        await self._initialize_fast()
        async with stdio_server() as (read_stream, write_stream):
            asyncio.create_task(self._initialize_slow())
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

        app = Starlette(
            routes=[
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
        logger.info("Received shutdown signal")
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