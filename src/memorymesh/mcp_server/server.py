import asyncio
import signal
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..config import AppConfig
from ..router import RouterClient
from ..hooks import hooks as global_hooks
from ..memory.hybrid_backend import HybridBackend
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..logging_ import setup_logging
from .tools import TOOLS
from .handlers import ToolHandlers

logger = logging.getLogger(__name__)

class MemoryMeshServer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.backend = HybridBackend(config)
        self.router = RouterClient(config.router)
        self.manager = MemoryManager(config, self.backend, self.router, hooks=global_hooks)
        self.session_store = SessionStore(config.session.db_path)
        self.handlers = ToolHandlers(self.manager, self.session_store)
        self.mcp_server = Server("memorymesh")
        self._register_tools()

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

    async def run_stdio(self):
        await self.backend.initialize()
        await self.session_store.initialize()
        if self.config.session.auto_create_session:
            active = await self.session_store.get_active_session(self.config.default_user_id)
            if active:
                session_id = active["session_id"]
                logger.info("Resuming active session: %s", session_id)
            else:
                session_id = await self.session_store.create_session(self.config.default_user_id)
                logger.info("Auto-created session: %s", session_id)
            await self.handlers.set_session(session_id)
            if self.config.session.auto_scan_codebase:
                await self.handlers._auto_scan_codebase(user_id=self.config.default_user_id)
            if self.config.session.auto_recall_on_start:
                await self.handlers._auto_recall_context(user_id=self.config.default_user_id)
        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.run(
                read_stream,
                write_stream,
                self.mcp_server.create_initialization_options(),
            )

    def run(self):
        loop = asyncio.get_event_loop()
        try:
            asyncio.run(self.run_stdio())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            asyncio.run(self.backend.close())
            asyncio.run(self.session_store.close())

def main():
    config = AppConfig.from_env()
    config.validate()
    setup_logging(config.log_level)
    server = MemoryMeshServer(config)
    server.run()