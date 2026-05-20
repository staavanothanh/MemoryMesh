import logging
from ..hooks import hooks as global_hooks
from ..memory.manager import MemoryManager
from ..errors import MemoryMeshError

logger = logging.getLogger(__name__)

class ToolHandlers:
    def __init__(self, manager: MemoryManager):
        self.manager = manager

    async def handle_remember(self, args: dict) -> dict:
        try:
            memory_id = await self.manager.add_memory(
                text=args["content"],
                tags=args.get("tags"),
                importance=args.get("importance", 3),
                user_id=args.get("user_id"),
            )
            return {"status": "success", "data": {"id": memory_id}}
        except MemoryMeshError as e:
            logger.error("Remember failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_recall(self, args: dict) -> dict:
        try:
            results = await self.manager.search_memory(
                query=args["query"],
                top_k=args.get("top_k", 5),
                user_id=args.get("user_id"),
            )
            return {"status": "success", "data": results}
        except MemoryMeshError as e:
            logger.error("Recall failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_forget(self, args: dict) -> dict:
        try:
            success = await self.manager.forget_memory(args["memory_id"])
            return {"status": "success", "data": {"id": args["memory_id"], "deleted": success}}
        except MemoryMeshError as e:
            logger.error("Forget failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_list_memories(self, args: dict) -> dict:
        try:
            results = await self.manager.list_memories(
                limit=args.get("limit", 100),
                offset=args.get("offset", 0),
                user_id=args.get("user_id"),
            )
            return {"status": "success", "data": results}
        except MemoryMeshError as e:
            logger.error("List failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_ping(self, args: dict) -> dict:
        return {"status": "success", "data": "pong"}