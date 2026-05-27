import time
from typing import Optional, Any


class ConversationTracker:
    CHOKE_THRESHOLD = 5
    READ_ONLY_TOOLS = frozenset({
        "recall", "remember", "list_memories", "ping",
        "list_sessions", "get_session_context",
    })

    def __init__(self, session_id: str, choke_point_enabled: bool = True):
        self.session_id = session_id
        self.choke_point_enabled = choke_point_enabled
        self._uncommitted_actions: int = 0
        self._hostage_data: Optional[Any] = None
        self._tool_footprints: list[str] = []
        self._last_activity: float = time.monotonic()
        self._has_unsaved_context: bool = False

    def record_tool_call(self, name: str, args: dict):
        self._last_activity = time.monotonic()
        self._has_unsaved_context = True

        query = args.get("query") or args.get("content") or args.get("user_message") or ""
        if query and name in ("recall", "remember", "commit_milestone"):
            self._tool_footprints.append(f"[{name}:{str(query)[:50]}]")
        else:
            self._tool_footprints.append(f"[{name}]")

        if name not in self.READ_ONLY_TOOLS and name != "commit_milestone":
            self._uncommitted_actions += 1

    def on_milestone_commit(self):
        self._uncommitted_actions = 0
        self._has_unsaved_context = False
        self._tool_footprints.clear()

    def engage_choke_point(self, data_to_hold: Any) -> str:
        if not self.choke_point_enabled:
            return ""
        self._hostage_data = data_to_hold
        return (
            "🛑 CHOKE-POINT ENGAGED: Your working memory is overloaded "
            f"({self._uncommitted_actions} uncommitted actions). "
            "I have the data you requested, but you MUST call `commit_milestone` first. "
            "Summarize your completed tasks now, and I will release the data."
        )

    def resolve_milestone(self) -> Optional[Any]:
        self._uncommitted_actions = 0
        self._has_unsaved_context = False
        self._tool_footprints.clear()
        released = self._hostage_data
        self._hostage_data = None
        return released

    def should_flush(self, idle_threshold: float = 60.0) -> bool:
        idle = time.monotonic() - self._last_activity > idle_threshold
        return idle and self._uncommitted_actions > 0

    def flush(self) -> str:
        footprint = " → ".join(self._tool_footprints) if self._tool_footprints else "[no tools]"
        self._tool_footprints.clear()
        return (
            f"[AUTO-SNAPSHOT] Uncommitted work (idle timeout). "
            f"Tool sequence: {footprint}"
        )

    def teardown_flush(self) -> Optional[str]:
        if not self._has_unsaved_context and self._uncommitted_actions == 0:
            return None
        footprint = " → ".join(self._tool_footprints) if self._tool_footprints else "[no tools]"
        self._has_unsaved_context = False
        self._tool_footprints.clear()
        self._uncommitted_actions = 0
        return (
            f"[SESSION-FINAL-SNAPSHOT] Session ended with unsaved context. "
            f"Tool sequence: {footprint}"
        )
