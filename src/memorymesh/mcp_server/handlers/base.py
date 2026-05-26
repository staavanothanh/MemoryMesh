import json
import asyncio
import time
import os
import logging
from typing import Optional, Any
from cachetools import TTLCache

from ._core import _log_bg, _safe_error_response, _ERROR_PRESERVE_KEYWORDS, _BOOTSTRAP_MAX_CHARS, _session_var, _client_name_var, _MAGENTA, _CYAN, _GREEN, _RESET, logger
from .semantic_filter import SemanticFilter
from .tracker import ConversationTracker
from ...hooks import hooks as global_hooks
from ...memory.manager import MemoryManager
from ...memory.session_store import SessionStore
from ...memory.fact_extractor import FactExtractor
from ...scanner import CodebaseScanner
from ...errors import MemoryMeshError
from ...utils.json_parser import clean_and_parse_llm_json
from ...utils.path_sanitizer import sanitize_workspace_path
from ...prompts import COMBINED_AGENT_INSTRUCTION, PERMANENT_LOG_DIRECTIVE, BOOTSTRAP_SNAPSHOT_PROMPT


class ToolHandlers:
    _MAX_BATCH_SIZE = 3
    _BOOTSTRAP_QUERIES = [
        "workspace state project bootstrap",
        "what did we do last session discussion topic",
    ]

    @property
    def _current_session_id(self):
        return _session_var.get()

    @_current_session_id.setter
    def _current_session_id(self, value):
        _session_var.set(value)

    def __init__(self, manager: MemoryManager, session_store: SessionStore):
        self.manager = manager
        self.session_store = session_store
        self._cached_workspace = None
        self._exchange_unsaved: bool = False
        self._reminder_count: int = 0
        self._fact_extractor = FactExtractor(manager.config, manager.router)
        self._fact_batch_buffer: list[str] = []
        self._fact_batch_lock = asyncio.Lock()
        self._global_bootstrap_ram_cache: dict[str, str] = {}
        self._recall_results_cache: dict[str, list] = {}
        self._last_depth_check_time: float = 0.0
        self._trackers: TTLCache = TTLCache(maxsize=500, ttl=7200)
        self._tracker_lock = asyncio.Lock()
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._write_worker_task: Optional[asyncio.Task] = None

    # ── Session management ──────────────────────────────────────────────

    async def set_session(self, session_id: str):
        self._current_session_id = session_id
        self._cached_workspace = None

    async def get_current_session_id(self) -> str:
        return self._current_session_id

    def consume_reminder(self) -> str:
        return ""

    async def _get_tracker(self, session_id: str) -> ConversationTracker:
        async with self._tracker_lock:
            if session_id not in self._trackers:
                self._trackers[session_id] = ConversationTracker(session_id)
            return self._trackers[session_id]

    async def _record_tool_call(self, session_id: str, name: str, args: dict):
        if not session_id:
            return
        try:
            tracker = await self._get_tracker(session_id)
            tracker.record_tool_call(name, args)
        except Exception:
            pass

    async def _on_milestone_commit(self, session_id: str):
        if not session_id:
            return
        try:
            tracker = await self._get_tracker(session_id)
            tracker.on_milestone_commit()
        except Exception:
            pass

    # ── Write worker ────────────────────────────────────────────────────

    def start_write_worker(self):
        if self._write_worker_task is None:
            self._write_worker_task = asyncio.create_task(self._write_worker())

    async def stop_write_worker(self):
        if self._write_worker_task:
            self._write_worker_task.cancel()
            try:
                await self._write_worker_task
            except asyncio.CancelledError:
                pass
            self._write_worker_task = None

    async def _write_worker(self):
        while True:
            try:
                task = await self._write_queue.get()
                await self.manager.add_memory(**task)
                self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Write worker error: %s", e)

    async def _idle_flush_loop(self):
        while True:
            await asyncio.sleep(15)
            try:
                async with self._tracker_lock:
                    session_ids = list(self._trackers.keys())
                for sid in session_ids:
                    async with self._tracker_lock:
                        tracker = self._trackers.get(sid)
                    if tracker and tracker.should_flush(60.0):
                        snapshot_text = tracker.flush()
                        logger.warning("Idle flush session %s — auto-snapshot", sid[:8])
                        if SemanticFilter.is_valuable(snapshot_text):
                            try:
                                self._write_queue.put_nowait({
                                    "text": snapshot_text,
                                    "tags": ["auto_save", "auto_snapshot", f"session:{sid}"],
                                    "importance": 2,
                                    "level": "session",
                                    "user_id": self.manager.config.default_user_id,
                                })
                            except asyncio.QueueFull:
                                logger.warning("Idle flush queue full, dropping session %s", sid[:8])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Idle flush loop error: %s", e)

    # ── Context & auto-save ─────────────────────────────────────────────

    async def _get_workspace_path(self) -> str:
        if self._cached_workspace is None and self._current_session_id:
            session = await self.session_store.get_session(self._current_session_id)
            if session:
                self._cached_workspace = session.get("workspace_path", "") or ""
            else:
                self._cached_workspace = ""
        return self._cached_workspace or ""

    async def _get_context_delta(self, session_id: str, uid: str, wp: str, max_tokens: int = 800) -> str:
        """Build a compressed Context Delta (<800 tokens) for auto-recall.
        
        Queries: last Milestone summary, recent context log, recent file paths.
        """
        try:
            parts = []

            # 1. Last Milestone from bootstrap cache or tag lookup
            cache_key = f"{uid}:{wp}" if wp else uid
            cached = self._global_bootstrap_ram_cache.get(cache_key)
            if cached:
                parts.append(f"[Previous Session]\n{cached[:400]}")
            else:
                tagged = await self.manager.backend.list_by_tag(uid, "milestone")
                for m in tagged[:3]:
                    mem_wp = m.get("metadata", {}).get("workspace_path", "")
                    if wp and mem_wp and mem_wp != wp:
                        continue
                    text = m.get("content", "")[:300]
                    if text:
                        parts.append(f"[Last Milestone]\n{text}")

            # 2. Recent context log (last 5 entries)
            context = await self.session_store.get_context_log(session_id, limit=5)
            if context:
                ctx_lines = []
                for e in context:
                    role = e.get("role", "?")
                    content = e.get("content", "")[:100]
                    ctx_lines.append(f"{role}: {content}")
                parts.append("[Recent Activity]\n" + "\n".join(ctx_lines))

            # 3. Recent workspace files (from scanner or snapshot)
            snaphots = await self.session_store.get_workspace_snapshots(session_id, limit=1)
            for snap in snaphots:
                data = snap.get("snapshot_data", {})
                files = data.get("files", {})
                dirs = files.get("dirs", [])[:10]
                fnames = files.get("files", [])[:10]
                if dirs or fnames:
                    parts.append("[Active Files]\n" + ", ".join(dirs + fnames))

            result = "\n\n".join(parts)

            # Strict token truncation
            token_count = len(result) // 4
            if token_count > max_tokens:
                result = result[:max_tokens * 4] + "\n...[truncated]"
            return result
        except Exception as e:
            logger.warning("Context delta failed: %s", e)
            return ""

    async def ensure_session(self, args: dict) -> bool:
        """Auto-resume or create a session if none active. Returns True if initialized.
        Phase 5.1: Returns Context Delta (<800 tokens) for auto-recall."""
        if self._current_session_id:
            session = await self.session_store.get_session(self._current_session_id)
            if session and session.get("status") == "active":
                return False

        uid = args.get("user_id", self.manager.config.default_user_id)
        wp = args.get("workspace_path", "")

        if wp:
            try:
                sessions = await self.session_store.list_sessions(
                    uid, limit=5, status="ended", include_deleted=False
                )
                for s in sessions:
                    if s.get("workspace_path", "") == wp:
                        sid = s["session_id"]
                        self._current_session_id = sid
                        self._cached_workspace = None

                        # Phase 5.1: Compute Context Delta for auto-recall
                        delta = await self._get_context_delta(sid, uid, wp)
                        if delta:
                            cache_key = f"{uid}:{wp}" if wp else uid
                            existing = self._global_bootstrap_ram_cache.get(cache_key, "")
                            if existing:
                                self._global_bootstrap_ram_cache[cache_key] = f"{existing}\n\n[Context Delta]\n{delta}"
                            else:
                                self._global_bootstrap_ram_cache[cache_key] = delta

                        asyncio.create_task(self._warm_resume_cache(sid, uid, wp))
                        _log_bg("AutoSession", f"Resumed ended session {sid[:8]} for {wp}", emoji="")
                        return True
            except Exception:
                pass

        await self.handle_new_session({
            "user_id": uid,
            "workspace_path": wp,
            "system_prompt": "",
        })
        return True

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
            session_tag = self._session_tag()
            match tool_name:
                case "remember":
                    content = args.get("content", "")
                    text = f"[DECISION] {content[:300]}" if content else None
                    imp = 4
                case "recall":
                    query = args.get("query", "")
                    count = len(result.get("data", [])) if isinstance(result, dict) else 0
                    text = f"[RECALL] '{query[:100]}' → {count} results" if query else None
                    imp = 2
                case "commit_milestone":
                    summary = args.get("summary", "")
                    tasks = args.get("tasks_done", "")
                    text = f"[MILESTONE] {summary[:100]} | Tasks: {tasks[:150]}" if summary else "[MILESTONE]"
                    imp = 4
                case "new_session":
                    wp = args.get("workspace_path", "")
                    text = f"[SESSION_START] workspace={wp}" if wp else "[SESSION_START]"
                    imp = 3
                case "end_session":
                    text = "[SESSION_END]"
                    imp = 3
                case "save_workspace_context":
                    text = "[WORKSPACE_SNAPSHOT]"
                    imp = 3
                case "delete_session":
                    text = "[SESSION_DELETED]"
                    imp = 3
                case "preserve_session_memories":
                    text = "[PRESERVE_MEMORIES]"
                    imp = 3
                case "resume_session":
                    sid = args.get("session_id", "")
                    text = f"[RESUME_SESSION] {sid[:8]}" if sid else "[RESUME_SESSION]"
                    imp = 3
                case _:
                    user_text = (args.get("query") or args.get("content")
                                 or args.get("user_message") or "")
                    text = f"[{tool_name.upper()}] {user_text[:200]}" if user_text else None
                    imp = 2

            if not text or not SemanticFilter.is_valuable(text):
                return

            await self._write_queue.put_nowait({
                "text": text,
                "tags": ["auto_save", tool_name, session_tag],
                "importance": imp,
                "level": "session",
                "user_id": args.get("user_id", self.manager.config.default_user_id),
                "workspace_path": await self._get_workspace_path(),
            })
        except asyncio.QueueFull:
            logger.warning("Write queue full, dropping auto_save for %s", tool_name)
        except Exception as e:
            logger.warning("save_auto_tool_context failed: %s", e)

    async def _note_tool_call(self, tool_name: str, arguments: dict):
        session_id = self._current_session_id
        if session_id and tool_name != "commit_milestone":
            await self._record_tool_call(session_id, tool_name, arguments)

        now = time.monotonic()
        if now - self._last_depth_check_time >= 10.0:
            self._last_depth_check_time = now
            asyncio.create_task(self._layer3_depth_check())

    async def _trigger_teardown_snapshot(self):
        """Flush tracker's unsaved context to DB at session end or disconnect."""
        if not self._current_session_id:
            return
        try:
            tracker = await self._get_tracker(self._current_session_id)
            snapshot_text = tracker.teardown_flush()
            if snapshot_text and SemanticFilter.is_valuable(snapshot_text):
                await self.manager.add_memory(
                    text=snapshot_text,
                    tags=["auto_save", "session_final", "teardown", self._session_tag()],
                    importance=3,
                    level="session",
                    user_id=self.manager.config.default_user_id,
                    workspace_path=await self._get_workspace_path(),
                )
                await self._log_context("assistant", snapshot_text, "auto_save_teardown")
                logger.info("Teardown snapshot saved for session %s", self._current_session_id[:8])
        except Exception as e:
            logger.warning("Teardown snapshot failed: %s", e)

    # ── Depth & scan ────────────────────────────────────────────────────

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

    # ── Bootstrap logic ─────────────────────────────────────────────────

    @staticmethod
    def _truncate_log_text(log: list, session_id: str, max_chars: int = _BOOTSTRAP_MAX_CHARS) -> str:
        """Truncate session log to fit within max_chars, keeping most recent messages."""
        if len(log) < 3:
            return f"Session {session_id[:8]} ended with {len(log)} messages."

        lines = [f"{e['role']}: {e['content']}" for e in log]
        if sum(len(l) for l in lines) <= max_chars:
            return "\n".join(lines)

        kept = []
        acc = 0
        for e in reversed(log):
            line = f"{e['role']}: {e['content']}"
            if acc + len(line) > max_chars:
                break
            kept.append(line)
            acc += len(line)
        return "\n".join(reversed(kept))

    async def _create_bootstrap_snapshot(self, session_id: str, user_id: str = "") -> Optional[str]:
        """Create L1 bootstrap snapshot (~1k tokens) from session context and save as a bootstrap memory.
        Also saves the compact_summary as a standalone session summary.
        Returns the stored text for RAM caching, or None on failure.
        """
        uid = user_id or self.manager.config.default_user_id
        try:
            log = await self.session_store.get_context_log(session_id, limit=self.manager.config.session.compact_threshold)
            log_text = self._truncate_log_text(log, session_id)

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

    # ── Phase 5: Lifecycle Automation ────────────────────────────────────

    async def recover_orphaned_sessions(self, uid: str = ""):
        """Find sessions without milestones and auto-generate summaries.
        
        Phase 5.2: Lazy Summarization — runs as background task on startup
        or idle trigger. Queries 'ended' sessions, checks for missing
        milestones, and calls LLM to generate one.
        """
        user_id = uid or self.manager.config.default_user_id
        try:
            sessions = await self.session_store.list_sessions(
                user_id, limit=10, status="ended", include_deleted=False
            )
            for session in sessions:
                sid = session["session_id"]
                # Check if this session already has a milestone
                tagged = await self.manager.backend.list_by_tag(user_id, "milestone")
                has_milestone = any(
                    f"session:{sid}" in m.get("metadata", {}).get("tags", [])
                    for m in tagged
                )
                if has_milestone:
                    continue
                # Get context logs for orphaned session
                context = await self.session_store.get_context_log(sid, limit=15)
                if len(context) < 3:
                    continue
                log_text = "\n".join(
                    f"{e['role']}: {e['content'][:200]}" for e in context[-10:]
                )
                prompt = f"Summarize this session into a Milestone: summary, tasks done, next steps.\n\n{log_text}"
                try:
                    response = await self.manager.router.call_llm_background(prompt, json_mode=False)
                    await self.manager.add_memory(
                        text=f"[Auto-Milestone] {response[:500]}",
                        tags=["milestone", "auto_recovery", f"session:{sid}"],
                        importance=3,
                        level="knowledge",
                        user_id=user_id,
                        workspace_path=session.get("workspace_path", ""),
                    )
                    logger.info("Orphan recovery: milestone created for session %s", sid[:8])
                except Exception as e:
                    logger.warning("Orphan recovery LLM failed for %s: %s", sid[:8], e)
        except Exception as e:
            logger.error("Orphan recovery failed: %s", e, exc_info=True)

    # ── Memory handlers ─────────────────────────────────────────────────

    async def handle_remember(self, args: dict) -> dict:
        wp = await self._get_workspace_path()
        level = args.get("level", "user")
        importance = args.get("importance", 3)
        content = args["content"]
        if level == "session" and (
            importance >= 4 or any(kw in content.lower() for kw in _ERROR_PRESERVE_KEYWORDS)
        ):
            level = "knowledge"

        # Phase 7.3: Bulk ingestion — return immediately, compute embedding in background
        is_bulk = len(content) > 1000

        try:
            memory_id = await self.manager.add_memory(
                text=content,
                tags=args.get("tags"),
                importance=importance,
                level=level,
                user_id=args.get("user_id"),
                workspace_path=wp if wp else args.get("workspace_path"),
                background=is_bulk,
            )
            await self._log_context("assistant", f"Saved memory: {args['content'][:200]}", "remember", str(args))
            if is_bulk:
                return {"status": "queued", "data": {"id": memory_id, "message": "Memory queued — embedding in background"}}
            return {"status": "success", "data": {"id": memory_id}}
        except MemoryMeshError as e:
            logger.error("Remember failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _search_graph_parallel(self, query: str, uid: str, wp: str) -> str:
        if self.manager.graph is None:
            return ""
        try:
            query_words = query.lower().split()
            word_scores: dict = {}
            entities = await self.manager.graph.list_entities(uid, limit=50)
            for ent in entities:
                ename = ent["name"].lower()
                score = 0.0
                for w in query_words:
                    if w in ename or ename in w:
                        score += 1.0
                if score > 0:
                    word_scores[ent["name"]] = score
            if not word_scores:
                return ""

            best_entity = max(word_scores, key=word_scores.get)
            result = await self.manager.graph.query_graph(best_entity, uid, limit=10)
            if result.get("neighbors"):
                return self.manager.graph.format_xml_from_neighbors(result["entity"], result["neighbors"])
        except Exception as e:
            logger.debug("Graph search skipped: %s", e)
        return ""

    async def handle_recall(self, args: dict) -> dict:
        wp = args.get("workspace_path") or await self._get_workspace_path()
        uid = args.get("user_id", self.manager.config.default_user_id)
        cache_key = f"{uid}:{wp}" if wp else uid

        # Phase 6.2: Cursor-based pagination support
        raw_cursor = args.get("cursor")
        if raw_cursor:
            return await self._handle_recall_paginated(uid, wp, args, raw_cursor)

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

            # CHOKE-POINT: Block recall if too many uncommitted actions
            session_id = self._current_session_id
            if session_id:
                tracker = await self._get_tracker(session_id)
                if tracker._uncommitted_actions >= ConversationTracker.CHOKE_THRESHOLD:
                    warning_msg = tracker.engage_choke_point(results)
                    return {
                        "status": "success",
                        "data": [],
                        "formatted": warning_msg,
                        "meta": {"tier": "choked", "count": 0},
                    }

            # PHASE 4: Parallel Graph Search — run alongside vector search
            graph_task = asyncio.create_task(self._search_graph_parallel(args["query"], uid, wp))

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

            graph_xml = await graph_task
            if graph_xml:
                formatted.append(f"\n{graph_xml}")

            meta = {"tier": tier_used, "count": len(results)}
            if graph_xml:
                meta["graph"] = True

            return {
                "status": "success",
                "data": results,
                "formatted": "\n".join(formatted) if formatted else "No relevant memories found.",
                "meta": meta,
            }
        except MemoryMeshError as e:
            logger.error("Recall failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _handle_recall_paginated(self, uid: str, wp: str, args: dict, raw_cursor: str) -> dict:
        """Phase 6.2: Cursor-based pagination for deep recall pages."""
        try:
            import json as _json
            cursor_dict = _json.loads(raw_cursor) if isinstance(raw_cursor, str) else raw_cursor
        except (_json.JSONDecodeError, TypeError):
            cursor_dict = None

        cm = getattr(self.manager, "context_manager", None)
        if not cm:
            return {"status": "error", "error": "Context manager not initialized"}

        max_tokens = args.get("max_tokens") or self.manager.config.token_budget
        try:
            page = await cm.get_context_page(
                user_id=uid,
                max_tokens=max_tokens,
                cursor=cursor_dict,
            )
        except Exception as e:
            logger.error("Paginated recall failed: %s", e)
            return {"status": "error", "error": str(e)}

        results = page.get("results", [])
        next_cursor = page.get("next_cursor")
        has_more = page.get("has_more", False)

        _log_bg("Recall", f"Page={page.get('next_cursor', {}).get('page', 1) if next_cursor else '?'}, results={len(results)}, query='{args['query'][:80]}'", emoji="")
        await self._log_context("assistant", f"Paginated recall: {len(results)} results, has_more={has_more}", "recall", str(args))

        formatted = []
        for r in results:
            prefix = "FACT" if "atomic_fact" in r.get("metadata", {}).get("tags", []) else "MEM"
            formatted.append(f"[{prefix}] {r.get('content', '')} (score: {r.get('score', 0.0):.2f})")

        meta = {"count": len(results), "page": next_cursor.get("page", 1) if next_cursor else 1, "has_more": has_more}
        if next_cursor:
            meta["next_cursor"] = _json.dumps(next_cursor) if isinstance(raw_cursor, str) else next_cursor

        return {
            "status": "success",
            "data": results,
            "formatted": "\n".join(formatted) if formatted else "No more results.",
            "meta": meta,
        }

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

    async def handle_recall_raw(self, args: dict) -> dict:
        try:
            session_id = args.get("session_id", self._current_session_id)
            limit = args.get("limit", 50)
            offset = args.get("offset", 0)
            tool_name = args.get("tool_name", "")
            status_filter = args.get("status", "")

            if not session_id:
                return {"status": "error", "error": "No session specified"}

            if tool_name or status_filter:
                results = await self.session_store.search_raw_log(
                    tool_name=tool_name, status=status_filter, limit=limit
                )
            else:
                results = await self.session_store.get_raw_log(
                    session_id=session_id, limit=limit, offset=offset
                )

            formatted = []
            for r in results:
                formatted.append(
                    f"[{r['timestamp']}] {r['tool_name']} ({r['status']}, {r.get('execution_time_ms', 0):.1f}ms)"
                )

            return {
                "status": "success",
                "data": results,
                "formatted": "\n".join(formatted) if formatted else "No raw log entries found.",
                "meta": {"count": len(results)},
            }
        except Exception as e:
            logger.error("recall_raw failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_learn_session(self, args: dict) -> dict:
        if not self.manager.config.instinct.enabled:
            return {"status": "error", "error": "Instinct learning is disabled"}
        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No session specified"}

            raw_logs = await self.session_store.get_raw_log(session_id, limit=500)
            if not raw_logs:
                return {"status": "success", "data": {"message": "No tool call data to learn from", "learned": 0}}

            tool_sequence = [r["tool_name"] for r in reversed(raw_logs)]
            wp = await self._get_workspace_path()

            from collections import Counter
            seq_counter = Counter()
            for i in range(len(tool_sequence)):
                for j in range(i + 1, min(i + 4, len(tool_sequence))):
                    sub_seq = tool_sequence[i:j]
                    if len(sub_seq) >= 2:
                        seq_counter[" → ".join(sub_seq)] += 1

            learned = 0
            existing = await self.manager.instinct_store.get_active_instincts(user_id)
            existing_conditions = {json.dumps(e["condition"]) for e in existing}

            for seq_str, count in seq_counter.items():
                if count < 2 or learned >= 10:
                    continue
                tools = seq_str.split(" → ")
                condition = {"type": "workflow", "sequence": tools}
                if json.dumps(condition) in existing_conditions:
                    continue
                confidence = min(0.8, 0.2 + count * 0.2)
                await self.manager.instinct_store.add_instinct(
                    user_id=user_id,
                    condition=condition,
                    action={"type": "suggest_workflow", "sequence": tools},
                    confidence=round(confidence, 4),
                    workspace_path=wp,
                )
                learned += 1

            await self._log_context("assistant", f"Learned {learned} workflow patterns from session", "learn_session", str(args))
            return {"status": "success", "data": {"learned": learned, "message": f"Learned {learned} workflow patterns from session {session_id[:8]}"}}
        except Exception as e:
            logger.error("learn_session failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ── Knowledge Graph Handlers (Cognitive Operations) ──────────────────

    async def _ensure_graph(self) -> bool:
        if self.manager.graph is None:
            logger.warning("GraphStore not initialized")
            return False
        return True

    async def handle_create_entity(self, args: dict) -> dict:
        if not await self._ensure_graph():
            return {"status": "error", "error": "Knowledge Graph not initialized"}
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            props = None
            raw = args.get("properties")
            if raw:
                import json as _json
                try:
                    props = _json.loads(raw)
                except (_json.JSONDecodeError, TypeError):
                    props = {"raw": raw}
            entity_id = await self.manager.graph.create_entity(
                name=args["name"],
                user_id=user_id,
                entity_type=args.get("entity_type", "concept"),
                properties=props,
            )
            await self._log_context("assistant", f"Entity created: {args['name']}", "create_entity", str(args))
            return {"status": "success", "data": {"id": entity_id, "name": args["name"], "type": args.get("entity_type", "concept")}}
        except Exception as e:
            logger.error("create_entity failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_create_relation(self, args: dict) -> dict:
        if not await self._ensure_graph():
            return {"status": "error", "error": "Knowledge Graph not initialized"}
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)

            source = await self.manager.graph.get_entity_by_name(args["source"], user_id)
            if not source:
                source_id = await self.manager.graph.create_entity(
                    name=args["source"], user_id=user_id, entity_type="concept"
                )
            else:
                source_id = source["id"]

            target = await self.manager.graph.get_entity_by_name(args["target"], user_id)
            if not target:
                target_id = await self.manager.graph.create_entity(
                    name=args["target"], user_id=user_id, entity_type="concept"
                )
            else:
                target_id = target["id"]

            relation_id = await self.manager.graph.create_relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=args["relation_type"],
                user_id=user_id,
                weight=args.get("weight", 1.0),
            )
            await self._log_context("assistant", f"Relation: {args['source']} --{args['relation_type']}--> {args['target']}", "create_relation", str(args))
            return {"status": "success", "data": {"id": relation_id, "source": args["source"], "target": args["target"], "relation_type": args["relation_type"]}}
        except Exception as e:
            logger.error("create_relation failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_query_graph(self, args: dict) -> dict:
        if not await self._ensure_graph():
            return {"status": "error", "error": "Knowledge Graph not initialized"}
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            result = await self.manager.graph.query_graph(
                entity_name=args["entity_name"],
                user_id=user_id,
                limit=args.get("limit", 20),
            )
            xml_triplet = self.manager.graph.format_xml_from_neighbors(
                result.get("entity"), result.get("neighbors", [])
            ) if result.get("entity") else ""
            error_msg = result.get("error", "")
            formatted = xml_triplet if xml_triplet else error_msg
            return {
                "status": "success" if result.get("entity") else "error",
                "data": result,
                "formatted": formatted or "No graph data found.",
            }
        except Exception as e:
            logger.error("query_graph failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_trace_entity(self, args: dict) -> dict:
        if not await self._ensure_graph():
            return {"status": "error", "error": "Knowledge Graph not initialized"}
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            result = await self.manager.graph.trace_entity(
                entity_name=args["entity_name"],
                user_id=user_id,
                max_depth=args.get("max_depth", 3),
                max_relations=args.get("max_relations", 20),
            )
            xml_triplet = self.manager.graph.format_xml_triplet(result.get("relations", []))
            formatted = xml_triplet if xml_triplet else (result.get("error", "") or "No path found.")
            return {
                "status": "success" if result.get("entity") else "error",
                "data": result,
                "formatted": formatted,
            }
        except Exception as e:
            logger.error("trace_entity failed: %s", e)
            return {"status": "error", "error": str(e)}

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

    async def handle_commit_milestone(self, args: dict) -> dict:
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            summary = args.get("summary", "")
            tasks_done = args.get("tasks_done", "")
            next_steps = args.get("next_steps", "")
            combined = f"[Milestone] Summary: {summary}\nTasks: {tasks_done}\nNext: {next_steps}"
            await self._log_context("user", summary)
            await self._log_context("assistant", f"Milestone committed: {tasks_done[:200]}")

            # Save as milestone memory with high importance
            milestone_id = await self.manager.add_memory(
                text=combined,
                tags=["milestone", "narrative_thread", "checkpoint", self._session_tag()],
                importance=4,
                level="session",
                user_id=user_id,
                workspace_path=await self._get_workspace_path(),
            )

            # Resolve tracker: reset actions + release hostage data
            released_data = None
            session_id = self._current_session_id
            if session_id:
                try:
                    tracker = await self._get_tracker(session_id)
                    released_data = tracker.resolve_milestone()
                except Exception:
                    pass

            # Extract atomic facts in background
            if self.manager.config.session.auto_extract_facts:
                asyncio.create_task(self._extract_and_save_facts(combined, user_id))

            response_text = f"✅ Milestone committed. (ID: {milestone_id[:12]})"
            if released_data:
                response_text += f"\n\n🔓 [HOSTAGE RELEASED] Here is the data you requested earlier:\n{released_data}"

            return {"status": "success", "data": {"milestone_id": milestone_id, "session_id": session_id, "message": response_text}}
        except MemoryMeshError as e:
            logger.error("Commit milestone failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_save_context_pair(self, args: dict) -> dict:
        """DEPRECATED: Auto-maps to commit_milestone for backward compatibility."""
        try:
            user_id = args.get("user_id", self.manager.config.default_user_id)
            user_msg = args.get("user_message", "")
            asst_msg = args.get("assistant_message", "")
            await self._log_context("user", user_msg)
            await self._log_context("assistant", f"[LEGACY SAVE] {asst_msg[:200]}")

            mapped_args = {
                "summary": "Legacy auto-mapped commit",
                "tasks_done": asst_msg[:300],
                "next_steps": "Unknown (Legacy mode)",
                "user_id": user_id,
            }
            result = await self.handle_commit_milestone(mapped_args)
            if result.get("status") == "success":
                result["data"]["message"] = (
                    "✅ Saved via legacy tool. "
                    "⚠️ DEPRECATION: `save_context_pair` is replaced by `commit_milestone`. "
                    "Use `commit_milestone(summary, tasks_done, next_steps)` instead."
                )
            return result
        except Exception as e:
            logger.error("Legacy save_context_pair failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ── Session handlers ────────────────────────────────────────────────

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
        try:
            safe_path = sanitize_workspace_path(workspace_path) if workspace_path else ""
        except ValueError:
            return ""
        if not safe_path or not os.path.isdir(safe_path):
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
        dir_path = os.path.join(safe_path, cfg.instructions_dir)
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
            fpath = os.path.join(safe_path, fname)
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
        try:
            safe_path = sanitize_workspace_path(workspace_path) if workspace_path else ""
        except ValueError:
            return
        if not cfg.docs_sync_enabled or not safe_path or not os.path.isdir(safe_path):
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
            doc = await _read_doc(os.path.join(safe_path, fname))
            if doc:
                name, content = doc
                try:
                    await self.manager.add_memory(
                        text=f"[Docs: {name}]\n{content}",
                        tags=["docs", "project_docs", f"doc:{name.lower()}"],
                        importance=4,
                        level="session",
                        user_id=user_id,
                        workspace_path=safe_path,
                    )
                    saved += 1
                except Exception as e:
                    logger.debug("Doc sync failed for %s: %s", name, e)

        if saved:
            _log_bg("SessionStart", f"Synced {saved} doc file(s)", emoji="")

    async def _bg_new_session_tasks(self, user_id: str, workspace_path: str):
        """Non-blocking background tasks for new session: bootstrap, git, codebase scan, docs sync."""
        # Bootstrap cache pre-compute
        try:
            scaffold = await self._get_bootstrap_scaffold(user_id, workspace_path)
            if scaffold:
                cache_key = f"{user_id}:{workspace_path}" if workspace_path else user_id
                self._global_bootstrap_ram_cache[cache_key] = scaffold
                _log_bg("Bootstrap", f"Pre-computed and cached for {cache_key}", emoji="")
        except Exception as e:
            _log_bg("Bootstrap", f"Background caching failed: {e}", emoji="")

        # Git context (non-blocking, inline cache update)
        try:
            import subprocess, os
            wp = workspace_path or os.getcwd()
            if os.path.isdir(os.path.join(wp, ".git")):
                loop = asyncio.get_event_loop()
                log = await loop.run_in_executor(
                    None, lambda: subprocess.run(
                        ["git", "log", "--oneline", "-5"],
                        capture_output=True, text=True, timeout=5, cwd=wp,
                    )
                )
                ctx = f"Recent commits:\n{log.stdout.strip()}" if log.stdout else ""
                diff = await loop.run_in_executor(
                    None, lambda: subprocess.run(
                        ["git", "diff", "--stat"],
                        capture_output=True, text=True, timeout=5, cwd=wp,
                    )
                )
                if diff.stdout:
                    ctx += f"\nUncommitted changes:\n{diff.stdout.strip()[:300]}"
                if ctx:
                    cache_key = f"{user_id}:{workspace_path}" if workspace_path else user_id
                    existing = self._global_bootstrap_ram_cache.get(cache_key, "")
                    self._global_bootstrap_ram_cache[cache_key] = f"{existing}\n\n{ctx}" if existing else ctx
        except Exception:
            pass

        # Codebase scan & docs sync
        asyncio.create_task(self._auto_scan_codebase(workspace_path, user_id))
        asyncio.create_task(self._sync_docs_files(workspace_path, user_id))

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

            # Parallel embedding: system prompt + session start marker
            add_tasks = []
            if system_prompt:
                add_tasks.append(self.manager.add_memory(
                    text=f"[System Prompt] {system_prompt}",
                    tags=["system_prompt", "session"],
                    importance=5,
                    level="session",
                    user_id=user_id,
                    workspace_path=workspace_path,
                ))
            add_tasks.append(self.manager.add_memory(
                text=f"[SESSION_START] Session {session_id[:8]} opened",
                tags=["session_start", "bootstrap", "session", self._session_tag()],
                importance=5,
                level="session",
                user_id=user_id,
                workspace_path=workspace_path,
            ))
            mem_results = await asyncio.gather(*add_tasks, return_exceptions=True)
            memory_id = mem_results[0] if (system_prompt and mem_results and not isinstance(mem_results[0], Exception)) else None
            if not system_prompt:
                memory_id = None

            # Background operations (non-blocking)
            asyncio.create_task(self._bg_new_session_tasks(user_id, workspace_path))

            data = {
                "session_id": session_id,
                "memory_id": memory_id,
                "message": "New session created",
                "project_context": "Loading project context...",
                "recall_instruction": COMBINED_AGENT_INSTRUCTION,
                "save_instruction": COMBINED_AGENT_INSTRUCTION,
                "permanent_log_directive": PERMANENT_LOG_DIRECTIVE,
                "tool_registry_reminder": (
                    "ACTION-BASED CHECKPOINT: MemoryMesh tracks your uncommitted actions. "
                    "Use commit_milestone(summary, tasks_done, next_steps) when finishing "
                    "a logical block of work. Do NOT call after every response."
                ),
            }

            await self._log_context("assistant", f"New session created: {session_id}", "new_session", str(args))

            return {"status": "success", "data": data}
        except MemoryMeshError as e:
            logger.error("New session failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def handle_end_session(self, args: dict) -> dict:
        try:
            session_id = args.get("session_id", self._current_session_id)
            user_id = args.get("user_id", self.manager.config.default_user_id)
            if not session_id:
                return {"status": "error", "error": "No active session to end"}

            # TEARDOWN HOOK: flush unsaved context before finalizing
            await self._trigger_teardown_snapshot()

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

            session = await self.session_store.get_session(session_id)
            if not session:
                return {"status": "error", "error": f"Session {session_id} not found"}

            # Auto-end active session before deletion
            did_end = False
            if session.get("status") == "active":
                logger.warning("Session %s is active — auto-ending before delete", session_id)
                await self.handle_end_session({"session_id": session_id, "user_id": user_id})
                did_end = True

            # Step 1: Preserve important memories (copy to knowledge level before deletion)
            preserved = await self.manager.preserve_important_memories(session_id, user_id)

            # Step 2: Hard delete vector memories (memory.db)
            deleted = await self.manager.delete_memories_by_session(session_id, user_id)

            # Step 3: Hard delete session record + context_log + snapshots (sessions.db)
            await self.session_store.hard_delete_session(session_id)

            if session_id == self._current_session_id:
                self._current_session_id = ""
            msg = f"Session permanently deleted" + (f" (auto-ended before delete)" if did_end else "")
            await self._log_context("assistant", f"Session deleted: {session_id}", "delete_session", str(args))
            return {"status": "success", "data": {"session_id": session_id, "preserved_count": preserved, "deleted_count": deleted, "message": msg}}
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

    # ── Workspace handlers ──────────────────────────────────────────────

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
