import asyncio
import signal
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..config import AppConfig
from ..router import RouterClient
from ..memory.chroma_impl import ChromaMemoryBackend
from ..memory.manager import MemoryManager
from ..logging_ import setup_logging
from .tools import TOOLS
from .handlers import ToolHandlers

logger = logging.getLogger(__name__)

class MemoryMeshServer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.backend = ChromaMemoryBackend(config.chroma.db_path)
        self.router = RouterClient(config.router)
        self.manager = MemoryManager(config, self.backend, self.router)
        self.handlers = ToolHandlers(self.manager)
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
            }
            handler = handler_map.get(name)
            if not handler:
                raise ValueError(f"Unknown tool: {name}")
            result = await handler(arguments)
            return [TextContent(type="text", text=str(result))]

    async def run_stdio(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.run(
                read_stream,
                write_stream,
                self.mcp_server.create_initialization_options(),
            )

    def run(self):
        loop = asyncio.get_event_loop()
        # Graceful shutdown on Windows (cần tự bắt KeyboardInterrupt)
        try:
            asyncio.run(self.run_stdio())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")

def main():
    config = AppConfig.from_env()
    config.validate()
    setup_logging(config.log_level)
    server = MemoryMeshServer(config)
    server.run()