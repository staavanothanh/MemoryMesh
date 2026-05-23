import json
import asyncio
import time
import os
import logging
from typing import Optional, List
from ..hooks import hooks as global_hooks
from ..memory.manager import MemoryManager
from ..memory.session_store import SessionStore
from ..memory.fact_extractor import FactExtractor
from ..scanner import CodebaseScanner
from ..errors import MemoryMeshError
from ..utils.json_parser import clean_and_parse_llm_json
from ..prompts import COMBINED_AGENT_INSTRUCTION, PERMANENT_LOG_DIRECTIVE, BOOTSTRAP_SNAPSHOT_PROMPT

_ERROR_PRESERVE_KEYWORDS = frozenset({
    "fix", "bug", "error", "crash", "failure", "fail", "hotfix", "patch",
    "workaround", "root cause", "lỗi", "sửa", "debug",
})

_MAGENTA = "\033[1;35m"
_CYAN = "\033[1;36m"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"


def _log_bg(label: str, msg: str, emoji: str = ""):
    """ANSI-colored structured log for background operations."""
    logger.info("%s %s[%s]%s %s", emoji, _MAGENTA, label, _RESET, msg)

logger = logging.getLogger(__name__)

class ToolHandlers:
    _MAX_BATCH_SIZE = 3
    _BOOTSTRAP_QUERIES = [
        "workspace state project bootstrap",
        "what did we do last session discussion topic",
    ]

    _READ_ONLY_TOOLS = frozenset({"ping", "list_sessions", "list_memories", "get_session_context"})

    def __init__(self, manager: MemoryManager, session_store: SessionStore):
        self.manager = manager
        self.session_store = session_store
        self._current_session_id: str = ""
        self._cached_workspace = None
        self._exchange_unsaved: bool = False
        self._reminder_count: int = 3
        self._fact_extractor = FactExtractor(manager.config, manager.router)
        self._fact_batch_buffer: list[str] = []
        self._fact_batch_lock = asyncio.Lock()
        self._global_bootstrap_ram_cache: dict[str, str] = {}
        self._recall_results_cache: dict[str, list] = {}
        self._last_depth_check_time: float = 0.0

    async def set_session(self, session_id: str):
        self._current_session_id = session_id
        self._cached_workspace = None

    async def get_current_session_id(self) -> str:
        return self._current_session_id

    def consume_reminder(self) -> str:
        if self._reminder_count > 0:
            self._reminder_count -= 1
            return "⚠️ Then call save_context_pair(user_msg, asst_msg)\n"
        return ""

    async def _get_workspace_path(self) -> str:
        if self._cached_workspace is None and self._current_session_id:
            session = await self.session_store.get_session(self._current_session_id)
            if session:
                self._cached_workspace = session.get("workspace_path", "") or ""
            else:
                self._cached_workspace = ""
        return self._cached_workspace or ""

    async def _log_context(self, role: str, content: str, tool_name: str = "", tool_args: str = ""):
        if self._current_session_id:
            try:
                await self.session_store.log_context(self._current_session_id, role, content, tool_name, tool_args)
            except Exception as e:
                logger.warning("Context log failed: %s", e)

    def _session_tag(self) -> str:
        return f"session:{self._current_session_id}" if self._current_session_id else ""

    async def _save_context_memory(self, role: str, content: str, tool_name: str = ""):
        if not (self._current_session_id and role in ("user", "assistant") and content.strip()):
            return
        try:
            tags = ["conversation", "session", role, self._session_tag()]
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
            logger.warning("Context memory save failed: %s", e)

    async def save_auto_tool_context(self, tool_name: str, args: dict, result: dict):
        if not self._current_session_id:
            return
        try:
            user_text = args.get("query") or args.get("content") or args.get("user_message") or ""
            session_tag = self._session_tag()
            if user_text:
                combined = f"[{tool_name.upper()}] User: {user_text[:300]}"
                tags = ["conversation", "session", "auto_save", tool_name, session_tag]
            else:
                combined = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)[:300]
                tags = ["conversation", "session", "auto_save", tool_name, session_tag]
            await self.manager.add_memory(
                text=combined,
                tags=tags,
                importance=4 if user_text else 3,
                level="session",
                user_id=args.get("user_id", self.manager.config.default_user_id),
                workspace_path=await self._get_workspace_path(),
            )
            await self._log_context("assistant", combined, tool_name, str(args))
        except Exception as e:
            logger.warning("save_auto_tool_context failed: %s", e)

    def _note_tool_call(self, tool_name: str):
        if tool_name in self._READ_ONLY_TOOLS:
            return
        if tool_name == "save_context_pair":
            self._exchange_unsaved = False
        elif tool_name not in ("recall", "end_session"):
            self._exchange_unsaved = True

        now = time.monotonic()
        if now - self._last_depth_check_time >= 10.0:
            self._last_depth_check_time = now
            asyncio.create_task(self._layer3_depth_check())

    async def _trigger_layer2(self):
        if not self._current_session_id:
            return
        try:
            recent = await self.session_store.get_context_log(self._current_session_id, limit=10)
            summary_lines = []
            for entry in recent[-5:]:
                summary_lines.append(f"[{entry['role']}] {entry.get('content', '')[:200]}")
            context_summary = "\n".join(summary_lines)
            await self.manager.add_memory(
                text=f"[AUTO_SAVE:LAYER2_MISSED_SAVE_CONTEXT_PAIR]\nUnsaved exchange detected. Recent context:\n{context_summary}",
                tags=["conversation", "session", "auto_save", "layer2", self._session_tag()],
                importance=4,
                level="session",
                user_id=self.manager.config.default_user_id,
                workspace_path=await self._get_workspace_path(),
            )
            await self._log_context("assistant", "Layer 2 auto-save triggered (missed save_context_pair)", "auto_save_layer2")
            logger.info("Layer 2 triggered for session %s", self._current_session_id[:8])
        except Exception as e:
            logger.warning("Layer 2 auto-save failed: %s", e)

    async def _layer3_depth_check(self):
        if not self._current_session_id:
            return
        try:
            current_depth = await self.session_store.get_context_log_count(self._current_session_id)
            threshold = self.manager.config.session.compact_threshold
            if current_depth >= int(threshold * 0.8):
                asyncio.create_task(self._create_bootstrap_snapshot(self._current_session_id, self.manager.config.default_user_id))
                logger.info("Layer 3 depth check: %d entries >= 80%% of %d, creating snapshot", current_depth, threshold)
        except Exception as e:
            logger.warning("Layer 3 depth check failed: %s", e)

    async def _auto_scan_codebase(self, workspace_path: str = "", user_id: str = ""):
        try:
            path = workspace_path or self.manager.config.default_user_id
            import os
            if not path or path == self.manager.config.default_user_id:
                path = os.getcwd()
            scanner = CodebaseScanner(workspace_path=path)
            snapshot = await asyncio.to_thread(scanner.scan)
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

    async def _create_bootstrap_snapshot(self, session_id: str, user_id: str = "") -> Optional[str]:
        """Create L1 bootstrap snapshot (~1k tokens) from session context and save as a bootstrap memory.
        Also saves the compact_summary as a standalone session summary.
        Returns the stored text for RAM caching, or None on failure.
        """
        uid = user_id or self.manager.config.default_user_id
        try:
            log = await self.session_store.get_context_log(session_id, limit=self.manager.config.session.compact_threshold)
            if len(log) < 3:
                log_text = f"Session {session_id[:8]} ended with {len(log)} messages."
            else:
                lines = [f"{e['role']}: {e['content']}" for e in log]
                total = sum(len(l) for l in lines)
                if total > 15000:
                    kept = []; acc = 0
                    for e in reversed(log):
                        l = f"{e['role']}: {e['content']}"
                        if acc + len(l) > 15000: break
                        kept.append(l); acc += len(l)
                    lines = list(reversed(kept))
                log_text = "\n".join(lines)

            prompt = BOOTSTRAP_SNAPSHOT_PROMPT.format(log=log_text)
            try:
                response = await self.manager.router.call_llm_background(prompt, json_mode=True)
                data = clean_and_parse_llm_json(response)
            except Exception as e:
                logger.warning("LLM bootstrap snapshot failed, using fallback: %s", e)
                data = {
                    "narrative_summary": f"Session {session_id[:8]} ended with {len(log)} messages.",
                    "compact_summary": f"Session {session_id[:8]} ended with {len(log)} messages.",
                    "discussion_topic": "",
                    "work_done": "",
                    "architectural_decisions": "",
                    "last_milestone": f"Session {session_id[:8]} ended",
                    "next_steps": "",
                }

            fields = ("narrative_summary", "compact_summary", "discussion_topic", "work_done",
                      "architectural_decisions", "last_milestone", "next_steps")
            for key in fields:
                data.setdefault(key, "")

            narrative = data.get("narrative_summary", "").strip()
            compact = data.get("compact_summary", "").strip()
            details = []
            for key in fields[2:]:
                val = data.get(key, "").strip()
                if val:
                    details.append(f"\n\u25a0 {key.replace('_', ' ').title()}: {val[:200]}")
            memory_text = f"[Bootstrap] {narrative[:300]}" + "".join(details)
            searchable_prefix = "buoi truoc session ket thuc du an lam viec last "
            full_text = searchable_prefix + memory_text[:970]

            memory_id = await self.manager.add_memory(
                text=full_text,
                tags=["bootstrap", "workspace_state", "session_summary"],
                importance=5,
                level="knowledge",
                user_id=uid,
                workspace_path=await self._get_workspace_path(),
            )

            # Save compact_summary as standalone session summary memory
            if compact:
                await self.manager.add_memory(
                    text=f"[Session Summary] {compact[:1000]}",
                    tags=["session_summary", "compacted"],
                    importance=4,
                    level="knowledge",
                    user_id=uid,
                    workspace_path=await self._get_workspace_path(),
                )

            _log_bg("Bootstrap", f"Snapshot saved for session {session_id[:8]} -> memory={memory_id[:12]}", emoji="")
            return full_text
        except Exception as e:
            logger.warning("Bootstrap snapshot failed: %s", e)
            return None

    async def _flush_fact_buffer(self):
        """Flush buffered conversations to LLM in a single batched call."""
        uid = self.manager.config.default_user_id
        async with self._fact_batch_lock:
            if not self._fact_batch_buffer:
                return
            batch = self._fact_batch_buffer[:]
            self._fact_batch_buffer.clear()
        try:
            facts = await self._fact_extractor.extract_facts_batch(batch)
            for item in facts:
                fact_tags = item.get("tags", []) + ["atomic_fact", item.get("confidence", "medium")]
                relation = item.get("relation", "")
                if relation:
                    fact_tags.append(f"relation:{relation}")
                importance = item.get("importance", 3)
                await self.manager.add_memory(
                    text=item["fact"],
                    tags=fact_tags,
                    importance=importance,
                    level="knowledge",
                    user_id=uid,
                    workspace_path=await self._get_workspace_path(),
                )
            if facts:
                logger.info("Saved %d atomic facts from %d conversations (batched)", len(facts), len(batch))
        except Exception as e:
            logger.warning("Batch fact extraction failed: %s", e)

    async def _finalize_session(self, session_id: str, user_id: str):
        """Core lifecycle: mark ended + bootstrap.

        Idempotent — safe to call multiple times. Each sub-step has its own
        error handling so a single failure doesn't block the rest.
        """
        try:
            _log_bg("Finalize", f"Finalizing session {session_id[:12]}...", emoji="")
            await self.session_store.end_session(session_id)
            snapshot_text = await self._create_bootstrap_snapshot(session_id, user_id)
            wp = await self._get_workspace_path()
            cache_key = f"{user_id}:{wp}" if wp else user_id
            if snapshot_text:
                self._global_bootstrap_ram_cache[cache_key] = snapshot_text
            asyncio.create_task(self._prewarm_and_flush(cache_key, user_id, wp))
            _log_bg("Finalize", f"Session {session_id[:12]} finalized", emoji="")
        except Exception as e:
            logger.error("Session finalization failed for %s: %s", session_id, e)

    async def _prewarm_and_flush(self, cache_key: str, user_id: str, wp: str):
        """Non-blocking pre-warm search + fact buffer flush after session finalization."""
        try:
            results, _, _ = await self.manager.search_with_fallback(
                query="session summary important decisions next steps",
                top_k=5, user_id=user_id, workspace_path=wp,
                max_tokens=500,
            )
            if results:
                mem_summary = "\n".join(f"[MEM] {r['content'][:200]}" for r in results)
                existing = self._global_bootstrap_ram_cache.get(cache_key, "")
                self._global_bootstrap_ram_cache[cache_key] = f"{existing}\n\nPre-computed:\n{mem_summary}" if existing else mem_summary
                self._recall_results_cache[cache_key] = results
        except Exception:
            pass
        await self._flush_fact_buffer()

    async def _warm_resume_cache(self, session_id: str, uid: str, wp: str | None):
        """Pre-warm cache for resumed session — runs async, does not block."""
        try:
            results, _, _ = await self.manager.search_with_fallback(
                query="session summary context next steps",
                top_k=10, user_id=uid, workspace_path=wp,
            )
            cache_key = f"{uid}:{wp}" if wp else uid
            if results:
                summary = "\n".join(f"- {r['content'][:200]}" for r in results[:5])
                existing = self._global_bootstrap_ram_cache.get(cache_key, "")
                # Dedup: if "Recalled context:" already exists, replace content
                existing_clean = existing.split("\n\nRecalled context:")[0].strip()
                self._global_bootstrap_ram_cache[cache_key] = existing_clean + f"\n\nRecalled context:\n{summary}" if existing_clean else f"Recalled context:\n{summary}"
                self._recall_results_cache[cache_key] = results
        except Exception as e:
            logger.warning("Resume cache warm failed: %s", e)

    async def _extract_and_save_facts(self, conversation: str, user_id: str = ""):
        """Buffer conversation pair; flush to LLM in batch when threshold is reached."""
        uid = user_id or self.manager.config.default_user_id
        async with self._fact_batch_lock:
            self._fact_batch_buffer.append(conversation)
            if len(self._fact_batch_buffer) < self._MAX_BATCH_SIZE:
                return
        await self._flush_fact_buffer()

    async def handle_remember(self, args: dict) -> dict:
        wp = await self._get_workspace_path()
        level = args.get("level", "user")
        importance = args.get("importance", 3)
        content = args["content"]
        if level == "session" and (
            importance >= 4 or any(kw in content.lower() for kw in _ERROR_PRESERVE_KEYWORDS)
        ):
            level = "knowledge"
        try:
            memory_id = await self.manager.add_memory(
                text=content,
                tags=args.get("tags"),
                importance=importance,
                level=level,
                user_id=args.get("user_id"),
                workspace_path=wp if wp else args.get("workspace_path"),
            )
            await self._log_context("assistant", f"Saved memory: {args['content'][:200]}", "remember", str(args))
            return {"status": "success", "data": {"id": memory_id}}
        except MemoryMeshError as e:
            logger.error("Remember failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_recall(self, args: dict) -> dict:
        if self._exchange_unsaved:
            self._exchange_unsaved = False
            asyncio.create_task(self._trigger_layer2())

        wp = args.get("workspace_path") or await self._get_workspace_path()
        uid = args.get("user_id", self.manager.config.default_user_id)
        cache_key = f"{uid}:{wp}" if wp else uid
        try:
            # Check recall cache before ANN search
            cached_results = self._recall_results_cache.pop(cache_key, None)
            if cached_results:
                results = cached_results
                tier_used = "cache"
            else:
                results, tier_used, _ = await self.manager.search_with_fallback(
                    query=args["query"],
                    top_k=args.get("top_k", 10),
                    user_id=uid,
                    workspace_path=wp,
                    max_tokens=args.get("max_tokens"),
                )
            _log_bg("Recall", f"Tier={tier_used}, results={len(results)}, query='{args['query'][:80]}'", emoji="")
            await self._log_context("assistant", f"Recalled {len(results)} memories via {tier_used} for: {args['query'][:200]}", "recall", str(args))
            if results:
                await self._save_context_memory("assistant", f"Recalled {len(results)} memories via {tier_used} for: {args['query'][:200]}", "recall")

            # Always prepend bootstrap context from RAM cache or tag lookup
            bootstrap_text = ""
            cached = self._global_bootstrap_ram_cache.get(cache_key)
            if cached:
                bootstrap_text = cached
            else:
                try:
                    tagged = await self.manager.backend.list_by_tag(uid, "bootstrap")
                    for m in tagged:
                        mem_wp = m.get("metadata", {}).get("workspace_path", "")
                        if wp and mem_wp and mem_wp != wp:
                            continue
                        text = m.get("content", "")
                        if text:
                            bootstrap_text = self._format_bootstrap(text, "previous session")
                            break
                except Exception:
                    pass

            formatted = []
            if bootstrap_text:
                formatted.append(bootstrap_text)

            for r in results:
                tags = r.get("tags", []) or []
                if "atomic_fact" in tags:
                    prefix = "FACT"
                elif "narrative_thread" in tags:
                    prefix = "NARRATIVE"
                elif "bootstrap" in tags or "workspace_state" in tags:
                    prefix = "BOOTSTRAP"
                elif "session_summary" in tags:
                    prefix = "SUMMARY"
                else:
                    prefix = "MEM"
                formatted.append(f"[{prefix}] {r.get('content', '')} (score: {r.get('score', 0.0):.2f})")
            return {
                "status": "success",
                "data": results,
                "formatted": "\n".join(formatted) if formatted else "No relevant memories found.",
                "meta": {"tier": tier_used, "count": len(results)},
            }
        except MemoryMeshError as e:
            logger.error("Recall failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_forget(self, args: dict) -> dict:
        try:
            success = await self.manager.forget_memory(args["memory_id"])
            return {"status": "success", "data": {"id": args["memory_id"], "archived": success}}
        except MemoryMeshError as e:
            logger.error("Forget failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_archive_memory(self, args: dict) -> dict:
        try:
            success = await self.manager.archive_memory(args["memory_id"])
            return {"status": "success", "data": {"id": args["memory_id"], "archived": success}}
        except MemoryMeshError as e:
            logger.error("Archive failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_unarchive_memory(self, args: dict) -> dict:
        try:
            success = await self.manager.unarchive_memory(args["memory_id"])
            return {"status": "success", "data": {"id": args["memory_id"], "unarchived": success}}
        except MemoryMeshError as e:
            logger.error("Unarchive failed: %s", e)
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
        user_id = args.get("user_id", self.manager.config.default_user_id)
        try:
            memories = await self.manager.backend.list_all(user_id, limit=10000)
            memory_count = len(memories)
        except Exception:
            memory_count = -1
        try:
            await self.manager.backend.fts_search("health", user_id, limit=1)
            fts_ok = True
        except Exception:
            fts_ok = False
        return {
            "status": "success",
            "data": {
                "status": "ok",
                "memory_count": memory_count,
                "fts_connected": fts_ok,
            },
        }

    async def handle_save_system_prompt(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            system_prompt = args["system_prompt"]
            if COMBINED_AGENT_INSTRUCTION not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{COMBINED_AGENT_INSTRUCTION}"
            if PERMANENT_LOG_DIRECTIVE not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{PERMANENT_LOG_DIRECTIVE}"
            await self.session_store.update_system_prompt(self._current_session_id, system_prompt)
            memory_id = await self.manager.add_memory(
                text=f"[System Prompt] {system_prompt}",
                tags=["system_prompt", "session"],
                importance=5,
                level="session",
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
            )
            return {"status": "success", "data": {"session_id": self._current_session_id, "memory_id": memory_id, "recall_instruction": COMBINED_AGENT_INSTRUCTION}}
        except MemoryMeshError as e:
            logger.error("Save system prompt failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_save_context_pair(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            user_msg = args["user_message"]
            asst_msg = args["assistant_message"]
            combined = f"User: {user_msg}\nAssistant: {asst_msg}"
            await self._log_context("user", user_msg)
            await self._log_context("assistant", asst_msg)

            # Narrative thread — preserves full exchange for chronological recall
            narrative_id = await self.manager.add_memory(
                text=combined,
                tags=["narrative_thread", "conversation", "session", self._session_tag()],
                importance=3,
                level="session",
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
            )

            # Layer 1: mark exchange as saved
            self._exchange_unsaved = False

            # Extract atomic facts in background (non-blocking)
            if self.manager.config.session.auto_extract_facts:
                asyncio.create_task(self._extract_and_save_facts(combined, user_id))

            return {"status": "success", "data": {"narrative_id": narrative_id, "session_id": self._current_session_id}}
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

            if len(context) < 3:
                return {
                    "status": "success",
                    "data": {
                        "session": session,
                        "context_log": [],
                        "formatted": f"### Session Context: {session_id[:8]}\n*Status: {session.get('status', 'unknown')}*\n*Empty session — no conversation history stored.*",
                    },
                }

            lines = [
                f"### Session Context: {session_id[:8]}",
                f"*Status: {session.get('status', 'unknown')}*",
                "",
            ]
            for msg in context[:limit]:
                role = msg["role"]
                content = msg["content"][:500]
                lines.append(f"**{role}**: {content}")
            markdown = "\n".join(lines)

            return {
                "status": "success",
                "data": {
                    "session": session,
                    "context_log": context,
                    "formatted": markdown,
                },
            }
        except Exception as e:
            logger.error("Get session context failed: %s", e)
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _format_bootstrap(text: str, source_label: str = "memory") -> str:
        return (
            f"\n=== PAST SESSION CONTEXT (from {source_label}) ===\n"
            f"{text[:800]}\n"
            f"======================================\n"
            f"COGNITIVE PROTOCOL \u2014 COLD START GROUND TRUTH:\n"
            f"1. This IS the verified ground truth of the last session.\n"
            f"2. Do NOT call git log or get_session_context to verify this data. Trust it implicitly.\n"
            f"3. Connect your next response directly to this context.\n"
            f"4. If the user references past work beyond this summary, call `recall(query)` for details."
        )

    async def _get_bootstrap_scaffold(self, user_id: str, workspace_path: str) -> str | None:
        """Hybrid 3-Layer Bootstrap Scaffolding — Zero-Blocking UX.

        Layer 1: Instant RAM Cache Lookup (~0ms)
        Layer 2: Fast Database Tag Lookup (~5ms, no embedding/LLM)
        Layer 3: Fallback Vector Search Pipeline (~2s, rarely reached)
        """
        uid = user_id or self.manager.config.default_user_id
        wp = workspace_path or await self._get_workspace_path()
        cache_key = f"{uid}:{wp}"

        # Layer 1: Instant RAM Cache Lookup (~0ms)
        cached = self._global_bootstrap_ram_cache.get(cache_key)
        if cached:
            _log_bg("Bootstrap", f"Layer 1 RAM cache hit for {uid}", emoji="")
            return cached

        # Layer 2: Fast Database Tag Lookup (~5ms) — no embedding/LLM
        try:
            tagged = await self.manager.backend.list_by_tag(uid, "bootstrap")
            for m in tagged:
                mem_wp = m.get("metadata", {}).get("workspace_path", "")
                if wp and mem_wp and mem_wp != wp:
                    continue
                text = m.get("content", "")
                if text:
                    _log_bg("Bootstrap", f"Layer 2 tag lookup hit for {uid}", emoji="")
                    formatted = self._format_bootstrap(text, "previous session")
                    self._global_bootstrap_ram_cache[cache_key] = formatted
                    return formatted
        except Exception as e:
            logger.warning("Layer 2 bootstrap tag lookup failed: %s", e)

        # Layer 3: Fallback Vector Search Pipeline — current search logic
        try:
            tasks = [
                self.manager.search_with_fallback(
                    query=q,
                    top_k=2,
                    user_id=uid,
                    workspace_path=wp,
                    max_tokens=800,
                    min_score_threshold=0.2,
                )
                for q in self._BOOTSTRAP_QUERIES
            ]
            all_results_raw = await asyncio.gather(*tasks, return_exceptions=True)

            seen_ids = set()
            merged = []
            best_tier = "empty"
            for result in all_results_raw:
                if isinstance(result, Exception):
                    continue
                results, tier, _ = result
                for r in results:
                    rid = r.get("id")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        merged.append(r)
                if results and tier != "empty":
                    best_tier = tier

            if not merged:
                return None

            best = next(
                (r for r in merged if
                 "bootstrap" in r.get("tags", []) or "session_summary" in r.get("tags", [])),
                merged[0],
            )
            source_label = {
                "semantic": "previous session",
                "fts_keyword": "keyword match",
                "chronological": "most recent activity",
                "empty": "none",
            }.get(best_tier, "memory")

            formatted = self._format_bootstrap(best.get("content", ""), source_label)
            self._global_bootstrap_ram_cache[cache_key] = formatted
            _log_bg("Bootstrap", f"Layer 3 search pipeline hit for {uid}", emoji="")
            return formatted
        except Exception as e:
            logger.warning("Layer 3 bootstrap search failed: %s", e)
            return None

    async def _load_session_instructions(self, workspace_path: str) -> str:
        cfg = self.manager.config.session.session_start
        if not workspace_path or not os.path.isdir(workspace_path):
            return ""

        loop = asyncio.get_event_loop()
        file_priority = cfg.instruction_file_priority or [
            "opencode.md", "CLAUDE.md", ".cursorrules", ".windsurfrules",
            "memorymesh.md", ".memorymesh.md",
        ]

        async def _read(path: str) -> str:
            try:
                if not os.path.isfile(path):
                    return ""
                size = os.path.getsize(path)
                if size > cfg.max_file_size:
                    _log_bg("SessionStart", f"Skipped {os.path.basename(path)} ({size}B exceeds {cfg.max_file_size}B)", emoji="")
                    return ""
                return await loop.run_in_executor(
                    None, lambda: _read_sync(path)
                )
            except Exception:
                return ""

        def _read_sync(path: str) -> str:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()

        parts = []

        # Tier 1: Global user-level file
        if cfg.global_instruction_file:
            gpath = os.path.expanduser(cfg.global_instruction_file)
            content = await _read(gpath)
            if content:
                parts.append(f"# Global Instructions\n{content}")

        # Tier 2: Instructions directory (all .md/.txt files, sorted)
        dir_path = os.path.join(workspace_path, cfg.instructions_dir)
        if os.path.isdir(dir_path):
            try:
                fnames = sorted(
                    f for f in os.listdir(dir_path)
                    if f.endswith(('.md', '.txt')) and os.path.isfile(os.path.join(dir_path, f))
                )
                for fname in fnames:
                    content = await _read(os.path.join(dir_path, fname))
                    if content:
                        parts.append(content)
            except Exception as e:
                logger.debug("Instructions dir scan failed: %s", e)

        # Tier 3: Auto-detect CLI/IDE files (first match wins)
        for fname in file_priority:
            fpath = os.path.join(workspace_path, fname)
            content = await _read(fpath)
            if content:
                parts.append(f"# From {fname}\n{content}")
                break

        result = "\n\n".join(parts) if parts else ""
        if result:
            _log_bg("SessionStart", f"Loaded {len(parts)} instruction source(s)", emoji="")
        return result

    async def _sync_docs_files(self, workspace_path: str, user_id: str):
        cfg = self.manager.config.session.session_start
        if not cfg.docs_sync_enabled or not workspace_path or not os.path.isdir(workspace_path):
            return

        loop = asyncio.get_event_loop()
        docs_files = cfg.docs_sync_files or ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "Makefile"]

        async def _read_doc(path: str) -> Optional[tuple]:
            try:
                if not os.path.isfile(path):
                    return None
                size = os.path.getsize(path)
                if size > cfg.max_file_size:
                    return None
                content = await loop.run_in_executor(
                    None, lambda: _read_doc_sync(path)
                )
                if not content:
                    return None
                return (os.path.basename(path), content)
            except Exception:
                return None

        def _read_doc_sync(path: str) -> str:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()[:2000]

        saved = 0
        for fname in docs_files:
            doc = await _read_doc(os.path.join(workspace_path, fname))
            if doc:
                name, content = doc
                try:
                    await self.manager.add_memory(
                        text=f"[Docs: {name}]\n{content}",
                        tags=["docs", "project_docs", f"doc:{name.lower()}"],
                        importance=4,
                        level="session",
                        user_id=user_id,
                        workspace_path=workspace_path,
                    )
                    saved += 1
                except Exception as e:
                    logger.debug("Doc sync failed for %s: %s", name, e)

        if saved:
            _log_bg("SessionStart", f"Synced {saved} doc file(s)", emoji="")

    async def handle_new_session(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            system_prompt = args.get("system_prompt", "")

            # Load session instructions from files (prepend to system prompt)
            workspace_path = args.get("workspace_path", "")
            instructions = await self._load_session_instructions(workspace_path)
            if instructions:
                system_prompt = f"{instructions}\n\n{system_prompt}"

            # Inject mandatory directives into system prompt
            if COMBINED_AGENT_INSTRUCTION not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{COMBINED_AGENT_INSTRUCTION}"
            if PERMANENT_LOG_DIRECTIVE not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{PERMANENT_LOG_DIRECTIVE}"
            if self._current_session_id:
                prev_sid = self._current_session_id
                self._current_session_id = ""
                asyncio.create_task(self._finalize_session(prev_sid, user_id))
            session_id = await self.session_store.create_session(
                user_id=user_id,
                system_prompt=system_prompt,
                workspace_path=workspace_path,
                auto_close_stale=True,
                stale_minutes=self.manager.config.session.stale_session_minutes,
            )
            self._current_session_id = session_id
            self._exchange_unsaved = False
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

            # Seed session start marker
            await self.manager.add_memory(
                text=f"[SESSION_START] Session {session_id[:8]} opened",
                tags=["session_start", "bootstrap", "session", self._session_tag()],
                importance=5,
                level="session",
                user_id=user_id,
                workspace_path=workspace_path,
            )

            # Compute bootstrap in background (non-blocking) so cache is ready for first recall
            async def _cache_bootstrap():
                try:
                    scaffold = await self._get_bootstrap_scaffold(user_id, workspace_path)
                    if scaffold:
                        cache_key = f"{user_id}:{workspace_path}" if workspace_path else user_id
                        self._global_bootstrap_ram_cache[cache_key] = scaffold
                        _log_bg("Bootstrap", f"Pre-computed and cached for {cache_key}", emoji="")
                except Exception as e:
                    _log_bg("Bootstrap", f"Background caching failed: {e}", emoji="")
            asyncio.create_task(_cache_bootstrap())

            # Extract project context from git for immediate model awareness
            project_context = ""
            loop = asyncio.get_event_loop()
            try:
                import subprocess, os
                wp = workspace_path or os.getcwd()
                if os.path.isdir(os.path.join(wp, ".git")):
                    log = await loop.run_in_executor(
                        None, lambda: subprocess.run(
                            ["git", "log", "--oneline", "-5"],
                            capture_output=True, text=True, timeout=5, cwd=wp,
                        )
                    )
                    if log.stdout:
                        project_context = f"Recent commits:\n{log.stdout.strip()}"
                    diff = await loop.run_in_executor(
                        None, lambda: subprocess.run(
                            ["git", "diff", "--stat"],
                            capture_output=True, text=True, timeout=5, cwd=wp,
                        )
                    )
                    if diff.stdout:
                        project_context += f"\nUncommitted changes:\n{diff.stdout.strip()[:300]}"
            except Exception:
                pass

            data = {
                "session_id": session_id,
                "memory_id": memory_id,
                "message": "New session created",
                "project_context": project_context or "MemoryMesh MCP server — memory mesh system",
                "recall_instruction": COMBINED_AGENT_INSTRUCTION,
                "save_instruction": COMBINED_AGENT_INSTRUCTION,
                "permanent_log_directive": PERMANENT_LOG_DIRECTIVE,
                "tool_registry_reminder": (
                    "MANDATORY: Re-read all tool descriptions (Fat Description) "
                    "to remember mandatory tools: save_context_pair, recall, remember. "
                    "This prevents forgetting in long conversations."
                ),
            }

            asyncio.create_task(self._auto_scan_codebase(workspace_path, user_id))
            asyncio.create_task(self._sync_docs_files(workspace_path, user_id))
            await self._log_context("assistant", f"New session created: {session_id}", "new_session", str(args))

            return {"status": "success", "data": data}
        except MemoryMeshError as e:
            logger.error("New session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_end_session(self, args: dict) -> dict:
        if self._exchange_unsaved:
            self._exchange_unsaved = False
            await self._trigger_layer2()

        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No active session to end"}

            await self._finalize_session(session_id, user_id)

            if session_id == self._current_session_id:
                self._current_session_id = ""
            await self._log_context("assistant", f"Session ended: {session_id}", "end_session", str(args))
            return {"status": "success", "data": {"session_id": session_id, "message": "Session ended"}}
        except MemoryMeshError as e:
            logger.error("End session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_delete_session(self, args: dict) -> dict:
        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No session specified"}
            preserved = await self.manager.preserve_important_memories(session_id, user_id)
            deleted = await self.manager.delete_memories_by_session(session_id, user_id)
            await self.session_store.mark_deleted(session_id)
            if session_id == self._current_session_id:
                self._current_session_id = ""
            await self._log_context("assistant", f"Session deleted: {session_id}", "delete_session", str(args))
            return {"status": "success", "data": {"session_id": session_id, "preserved_count": preserved, "deleted_count": deleted, "message": "Session deleted"}}
        except MemoryMeshError as e:
            logger.error("Delete session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_preserve_session_memories(self, args: dict) -> dict:
        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No session specified"}
            preserved = await self.manager.preserve_important_memories(session_id, user_id)
            return {"status": "success", "data": {"session_id": session_id, "preserved_count": preserved, "message": f"Preserved {preserved} memories"}}
        except MemoryMeshError as e:
            logger.error("Preserve session memories failed: %s", e)
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
                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(
                            None, lambda: subprocess.run(
                                ["git", "log", "--oneline", "-5"],
                                capture_output=True, text=True, timeout=5, cwd=workspace_path,
                            )
                        )
                        snapshot["git"]["recent_commits"] = result.stdout.strip().split("\n") if result.stdout else []
                    except Exception:
                        snapshot["git"]["recent_commits"] = []
                    try:
                        result = await loop.run_in_executor(
                            None, lambda: subprocess.run(
                                ["git", "status", "--short"],
                                capture_output=True, text=True, timeout=5, cwd=workspace_path,
                            )
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
            await self._log_context("assistant", f"Workspace snapshot saved", "save_workspace_context", str(args))
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

            # Layer 2: DB reads — fast (~5ms)
            context = await self.session_store.get_context_log(session_id, limit=50)
            snapshots = await self.session_store.get_workspace_snapshots(session_id)

            # Layer 1: instant from RAM cache
            wp = await self._get_workspace_path()
            cache_key = f"{user_id}:{wp}" if wp else user_id
            cached_bootstrap = self._global_bootstrap_ram_cache.get(cache_key, "")

            # Layer 3: cached recall results from pre-compute
            cached_results = self._recall_results_cache.pop(cache_key, None)
            if cached_results:
                recalled = [
                    {"id": m["id"], "content": m["content"][:300], "score": m["score"]}
                    for m in cached_results
                ]
            else:
                recalled = []
                # Fire background search — does not block response
                asyncio.create_task(self._warm_resume_cache(session_id, user_id, wp))

            await self._log_context("assistant", f"Resumed session {session_id}: {len(context)} messages, {len(recalled)} memories", "resume_session", str(args))
            await self._save_context_memory("assistant", f"Resumed session {session_id}: {len(context)} messages, {len(recalled)} memories", "resume_session")

            return {
                "status": "success",
                "data": {
                    "session": session,
                    "context_log": context,
                    "workspace_snapshots": snapshots,
                    "recalled_memories": recalled,
                    "bootstrap": cached_bootstrap,
                    "message": f"Restored session {session_id[:8]}..." + ("" if recalled else " Memories loading in background."),
                },
            }
        except MemoryMeshError as e:
            logger.error("Resume session failed: %s", e)
            return {"status": "error", "error": str(e)}