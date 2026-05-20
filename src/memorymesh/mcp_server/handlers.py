import logging
from ..hooks import hooks as global_hooks
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..errors import MemoryMeshError

logger = logging.getLogger(__name__)

class ToolHandlers:
    def __init__(self, manager: MemoryManager, session_store: SessionStore):
        self.manager = manager
        self.session_store = session_store
        self._current_session_id: str = ""

    async def set_session(self, session_id: str):
        self._current_session_id = session_id

    async def _auto_log(self, role: str, content: str, tool_name: str = "", tool_args: str = ""):
        if self._current_session_id:
            try:
                await self.session_store.log_context(self._current_session_id, role, content, tool_name, tool_args)
            except Exception as e:
                logger.warning("Auto-log failed: %s", e)

    async def handle_remember(self, args: dict) -> dict:
        try:
            memory_id = await self.manager.add_memory(
                text=args["content"],
                tags=args.get("tags"),
                importance=args.get("importance", 3),
                level=args.get("level", "user"),
                user_id=args.get("user_id"),
            )
            await self._auto_log("assistant", f"Saved memory: {args['content'][:200]}", "remember", str(args))
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
            await self._auto_log("assistant", f"Recalled {len(results)} memories for: {args['query'][:200]}", "recall", str(args))
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

    async def handle_save_system_prompt(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            system_prompt = args["system_prompt"]
            await self.session_store.update_system_prompt(self._current_session_id, system_prompt)
            memory_id = await self.manager.add_memory(
                text=f"[System Prompt] {system_prompt}",
                tags=["system_prompt", "session"],
                importance=5,
                level="session",
                user_id=user_id,
            )
            return {"status": "success", "data": {"session_id": self._current_session_id, "memory_id": memory_id}}
        except MemoryMeshError as e:
            logger.error("Save system prompt failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_save_context_pair(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            user_msg = args["user_message"]
            asst_msg = args["assistant_message"]
            combined = f"User: {user_msg}\nAssistant: {asst_msg}"
            await self._auto_log("user", user_msg)
            await self._auto_log("assistant", asst_msg)
            memory_id = await self.manager.add_memory(
                text=combined,
                tags=["conversation", "session"],
                importance=3,
                level="session",
                user_id=user_id,
            )
            return {"status": "success", "data": {"memory_id": memory_id, "session_id": self._current_session_id}}
        except MemoryMeshError as e:
            logger.error("Save context pair failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_list_sessions(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            limit = args.get("limit", 10)
            sessions = await self.session_store.list_sessions(user_id, limit)
            return {"status": "success", "data": sessions}
        except Exception as e:
            logger.error("List sessions failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_get_session_context(self, args: dict) -> dict:
        try:
            session_id = args["session_id"]
            limit = args.get("limit", 50)
            session = await self.session_store.get_session(session_id)
            if not session:
                return {"status": "error", "error": f"Session {session_id} not found"}
            context = await self.session_store.get_context_log(session_id, limit)
            return {"status": "success", "data": {"session": session, "context_log": context}}
        except Exception as e:
            logger.error("Get session context failed: %s", e)
            return {"status": "error", "error": str(e)}