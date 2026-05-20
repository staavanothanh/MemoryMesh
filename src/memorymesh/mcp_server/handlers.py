import json
import asyncio
import logging
from typing import Optional, List
from ..hooks import hooks as global_hooks
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..memory.fact_extractor import FactExtractor
from ..scanner import CodebaseScanner
from ..errors import MemoryMeshError
from ..prompts import RECALL_INSTRUCTION, SESSION_COMPACT_PROMPT

logger = logging.getLogger(__name__)

class ToolHandlers:
    def __init__(self, manager: MemoryManager, session_store: SessionStore):
        self.manager = manager
        self.session_store = session_store
        self._current_session_id: str = ""
        self._fact_extractor = FactExtractor(manager.config, manager.router)

    async def set_session(self, session_id: str):
        self._current_session_id = session_id
        self._cached_workspace = None

    async def get_current_session_id(self) -> str:
        return self._current_session_id

    async def _get_workspace_path(self) -> str:
        if not hasattr(self, '_cached_workspace'):
            self._cached_workspace = None
        if self._cached_workspace is None and self._current_session_id:
            session = await self.session_store.get_session(self._current_session_id)
            if session:
                self._cached_workspace = session.get("workspace_path", "") or ""
            else:
                self._cached_workspace = ""
        return self._cached_workspace or ""

    async def _auto_log(self, role: str, content: str, tool_name: str = "", tool_args: str = "", save_memory: bool = False):
        if self._current_session_id:
            try:
                await self.session_store.log_context(self._current_session_id, role, content, tool_name, tool_args)
                if save_memory and role in ("user", "assistant") and content.strip():
                    tags = ["conversation", "session", role]
                    if tool_name:
                        tags.append(tool_name)
                    await self.manager.add_memory(
                        text=f"[{role}] {content[:500]}",
                        tags=tags,
                        importance=4 if tool_name else 3,
                        level="session",
                        user_id=self.manager.config.default_user_id,
                        workspace_path=await self._get_workspace_path(),
                    )
            except Exception as e:
                logger.warning("Auto-log failed: %s", e)

    async def _auto_scan_codebase(self, workspace_path: str = "", user_id: str = ""):
        try:
            path = workspace_path or self.manager.config.default_user_id
            import os
            if not path or path == self.manager.config.default_user_id:
                path = os.getcwd()
            scanner = CodebaseScanner(workspace_path=path)
            snapshot = scanner.scan()
            summary = snapshot.get("summary", "")
            await self.session_store.save_workspace_snapshot(self._current_session_id, snapshot)
            memory_id = await self.manager.add_memory(
                text=f"[Codebase Snapshot] {summary}",
                tags=["codebase", "workspace", "knowledge"],
                importance=4,
                level="knowledge",
                user_id=user_id or self.manager.config.default_user_id,
                workspace_path=path,
            )
            logger.info("Codebase auto-scanned: %d entries, memory=%s", len(snapshot.get("tree", [])), memory_id)
        except Exception as e:
            logger.warning("Codebase auto-scan failed: %s", e)

    async def _auto_recall_context(self, user_id: str = "", max_items: int = 3):
        if not self._current_session_id:
            return
        uid = user_id or self.manager.config.default_user_id
        try:
            results = await self.manager.search_memory(
                query="session context project plan development",
                top_k=max_items,
                user_id=uid,
                level_filter=["knowledge", "user"],
                workspace_path=await self._get_workspace_path(),
            )
            if results:
                summary = "\n".join(f"- [{r['importance']}] {r['content'][:200]}" for r in results)
                await self.session_store.log_context(
                    self._current_session_id, "assistant",
                    f"[Auto-Recalled Context]\n{summary}",
                    "recall",
                )
                logger.info("Auto-recalled %d memories for session %s", len(results), self._current_session_id)
        except Exception as e:
            logger.warning("Auto-recall failed: %s", e)

    async def _compact_session(self, session_id: str, user_id: str = ""):
        uid = user_id or self.manager.config.default_user_id
        try:
            log = await self.session_store.get_context_log(session_id, limit=self.manager.config.session.compact_threshold)
            if len(log) < 3:
                return
            log_text = "\n".join(f"{entry['role']}: {entry['content'][:300]}" for entry in log)
            prompt = SESSION_COMPACT_PROMPT.format(log=log_text)
            summary = ""
            try:
                response = await self.manager.router.call_llm(prompt)
                summary = response.strip().strip('"').strip("'")
            except Exception as e:
                logger.warning("LLM compact failed, using fallback: %s", e)
                summary = f"Session {session_id}: {len(log)} messages, ended."
            if not summary:
                summary = f"Session {session_id}: {len(log)} messages, ended."
            memory_id = await self.manager.add_memory(
                text=f"[Session Summary] {summary[:1000]}",
                tags=["session_summary", "compacted"],
                importance=4,
                level="knowledge",
                user_id=uid,
                workspace_path=await self._get_workspace_path(),
            )
            logger.info("Session compacted: %s -> memory=%s", session_id, memory_id)
        except Exception as e:
            logger.warning("Session compaction failed: %s", e)

    async def _extract_and_save_facts(self, conversation: str, user_id: str = ""):
        """Extract atomic facts from conversation and save them as memories (background)."""
        uid = user_id or self.manager.config.default_user_id
        try:
            facts = await self._fact_extractor.extract_facts(conversation)
            for item in facts:
                fact_tags = item.get("tags", []) + ["atomic_fact", item.get("confidence", "medium")]
                importance = self._fact_extractor._confidence_to_importance(item.get("confidence", "medium"))
                await self.manager.add_memory(
                    text=item["fact"],
                    tags=fact_tags,
                    importance=importance,
                    level="knowledge",
                    user_id=uid,
                    workspace_path=await self._get_workspace_path(),
                )
            if facts:
                logger.info("Saved %d atomic facts from conversation", len(facts))
        except Exception as e:
            logger.warning("Atomic fact extraction background task failed: %s", e)

    async def handle_remember(self, args: dict) -> dict:
        wp = await self._get_workspace_path()
        try:
            memory_id = await self.manager.add_memory(
                text=args["content"],
                tags=args.get("tags"),
                importance=args.get("importance", 3),
                level=args.get("level", "user"),
                user_id=args.get("user_id"),
                workspace_path=wp if wp else args.get("workspace_path"),
            )
            await self._auto_log("assistant", f"Saved memory: {args['content'][:200]}", "remember", str(args))
            return {"status": "success", "data": {"id": memory_id}}
        except MemoryMeshError as e:
            logger.error("Remember failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_recall(self, args: dict) -> dict:
        wp = args.get("workspace_path") or await self._get_workspace_path()
        try:
            results = await self.manager.search_memory(
                query=args["query"],
                top_k=args.get("top_k", 10),
                user_id=args.get("user_id"),
                workspace_path=wp,
            )
            await self._auto_log("assistant", f"Recalled {len(results)} memories for: {args['query'][:200]}", "recall", str(args), save_memory=True)
            # Format as atomic fact bullets for clarity
            formatted = []
            for r in results:
                tags = r.get("tags", []) or []
                prefix = "📌" if "atomic_fact" in tags else "•"
                formatted.append(f"{prefix} {r['content']} (relevance: {r['score']:.2f})")
            return {
                "status": "success",
                "data": results,
                "formatted": "\n".join(formatted) if formatted else "No relevant memories found.",
            }
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
            if RECALL_INSTRUCTION not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{RECALL_INSTRUCTION}"
            await self.session_store.update_system_prompt(self._current_session_id, system_prompt)
            memory_id = await self.manager.add_memory(
                text=f"[System Prompt] {system_prompt}",
                tags=["system_prompt", "session"],
                importance=5,
                level="session",
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
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
            await self._auto_log("user", user_msg, save_memory=True)
            await self._auto_log("assistant", asst_msg, save_memory=True)
            memory_id = await self.manager.add_memory(
                text=combined,
                tags=["conversation", "session"],
                importance=3,
                level="session",
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
            )
            # Extract atomic facts in background (non-blocking)
            asyncio.create_task(self._extract_and_save_facts(combined, user_id))
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
            snapshots = await self.session_store.get_workspace_snapshots(session_id)
            return {"status": "success", "data": {"session": session, "context_log": context, "workspace_snapshots": snapshots}}
        except Exception as e:
            logger.error("Get session context failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_new_session(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            system_prompt = args.get("system_prompt", "")
            workspace_path = args.get("workspace_path", "")
            if self._current_session_id:
                await self.session_store.end_session(self._current_session_id)
            session_id = await self.session_store.create_session(
                user_id=user_id,
                system_prompt=system_prompt,
                workspace_path=workspace_path,
                auto_close_stale=True,
            )
            self._current_session_id = session_id
            if system_prompt:
                memory_id = await self.manager.add_memory(
                    text=f"[System Prompt] {system_prompt}",
                    tags=["system_prompt", "session"],
                    importance=5,
                    level="session",
                    user_id=user_id,
                    workspace_path=workspace_path,
                )
            else:
                memory_id = None
            asyncio.create_task(self._auto_scan_codebase(workspace_path, user_id))
            await self._auto_log("assistant", f"New session created: {session_id}", "new_session", str(args))
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "memory_id": memory_id,
                    "message": "Session mới đã được tạo",
                },
            }
        except MemoryMeshError as e:
            logger.error("New session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_end_session(self, args: dict) -> dict:
        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No active session to end"}
            if self.manager.config.session.auto_compact_on_end:
                await self._compact_session(session_id, user_id)
            await self.session_store.end_session(session_id)
            if session_id == self._current_session_id:
                self._current_session_id = ""
            await self._auto_log("assistant", f"Session ended: {session_id}", "end_session", str(args))
            return {"status": "success", "data": {"session_id": session_id, "message": "Session đã kết thúc"}}
        except MemoryMeshError as e:
            logger.error("End session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_save_workspace_context(self, args: dict) -> dict:
        try:
            import os
            import subprocess
            user_id = args.get("user_id", self.manager.config.default_user_id)
            workspace_path = args.get("workspace_path", "")
            if not workspace_path:
                session = await self.session_store.get_session(self._current_session_id)
                if session:
                    workspace_path = session.get("workspace_path", "")
            if not workspace_path:
                workspace_path = os.getcwd()
            snapshot = {"workspace_path": workspace_path, "files": [], "git": {}, "dependencies": {}}
            if os.path.isdir(workspace_path):
                root_dirs = [d for d in os.listdir(workspace_path) if os.path.isdir(os.path.join(workspace_path, d)) and not d.startswith((".", "__"))][:30]
                root_files = [f for f in os.listdir(workspace_path) if os.path.isfile(os.path.join(workspace_path, f))][:50]
                snapshot["files"] = {"dirs": root_dirs, "files": root_files}
                git_dir = os.path.join(workspace_path, ".git")
                if os.path.isdir(git_dir):
                    try:
                        result = subprocess.run(
                            ["git", "log", "--oneline", "-5"],
                            capture_output=True, text=True, timeout=5, cwd=workspace_path,
                        )
                        snapshot["git"]["recent_commits"] = result.stdout.strip().split("\n") if result.stdout else []
                    except Exception:
                        snapshot["git"]["recent_commits"] = []
                    try:
                        result = subprocess.run(
                            ["git", "status", "--short"],
                            capture_output=True, text=True, timeout=5, cwd=workspace_path,
                        )
                        snapshot["git"]["status"] = result.stdout.strip().split("\n") if result.stdout else []
                    except Exception:
                        snapshot["git"]["status"] = []
                pyproject = os.path.join(workspace_path, "pyproject.toml")
                if os.path.isfile(pyproject):
                    try:
                        with open(pyproject, "r", encoding="utf-8") as f:
                            content = f.read()
                        import re
                        deps = re.findall(r'^([\w\-]+)\s*=\s*["\']', content, re.MULTILINE)
                        snapshot["dependencies"]["project"] = deps[:30]
                    except Exception:
                        pass
            await self.session_store.save_workspace_snapshot(self._current_session_id, snapshot)
            memory_id = await self.manager.add_memory(
                text=f"[Workspace Snapshot] {json.dumps(snapshot, ensure_ascii=False)[:500]}",
                tags=["workspace", "session"],
                importance=4,
                level="session",
                user_id=user_id,
                workspace_path=workspace_path,
            )
            await self._auto_log("assistant", f"Workspace snapshot saved", "save_workspace_context", str(args))
            return {"status": "success", "data": {"memory_id": memory_id, "snapshot": snapshot}}
        except MemoryMeshError as e:
            logger.error("Save workspace context failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_resume_session(self, args: dict) -> dict:
        try:
            session_id = args["session_id"]
            top_k = args.get("top_k", 10)
            user_id = args.get("user_id", self.manager.config.default_user_id)

            session = await self.session_store.get_session(session_id)
            if not session:
                return {"status": "error", "error": f"Session {session_id} not found"}

            context = await self.session_store.get_context_log(session_id, limit=50)
            snapshots = await self.session_store.get_workspace_snapshots(session_id)

            recall_query = session.get("system_prompt", "") or f"session {session_id[:8]} context"
            memories = await self.manager.search_memory(
                query=recall_query,
                top_k=top_k,
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
            )

            await self._auto_log("assistant", f"Resumed session {session_id}: {len(context)} messages, {len(memories)} memories", "resume_session", str(args), save_memory=True)

            return {
                "status": "success",
                "data": {
                    "session": session,
                    "context_log": context,
                    "workspace_snapshots": snapshots,
                    "recalled_memories": [
                        {"id": m["id"], "content": m["content"][:300], "score": m["score"]}
                        for m in memories
                    ],
                    "message": f"Đã khôi phục session {session_id[:8]}...",
                },
            }
        except MemoryMeshError as e:
            logger.error("Resume session failed: %s", e)
            return {"status": "error", "error": str(e)}