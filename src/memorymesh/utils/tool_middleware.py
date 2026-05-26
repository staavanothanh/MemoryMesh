"""ToolExecutionMiddleware — sliding window + JIT instinct injection.

Maintains a deque of the last 5 tool calls. After each tool execution,
matches the context window against RAM-cached compiled instincts.
If a match is found, injects the reaction text into the tool output.
"""

import logging
from collections import deque
from typing import List, Optional

from ..memory.instinct_manager import InstinctManager

logger = logging.getLogger(__name__)


class ToolExecutionMiddleware:
    """Sliding window context + JIT instinct injection.

    Tracks the last 5 tool calls (memory-light: ~few hundred bytes).
    After each tool execution, evaluates the window against RAM-cached
    instincts and injects reactions into output if matched.
    """

    def __init__(self, instinct_manager: InstinctManager):
        self.instinct_manager = instinct_manager
        self._context_window: deque = deque(maxlen=5)
        self._tool_sequences: List[str] = []
        self._current_project_id: str = ""

    def set_project(self, project_id: str):
        self._current_project_id = project_id

    def record_call(self, tool_name: str, args: dict) -> Optional[List[str]]:
        """Record a tool call in the sliding window.

        Returns matched reactions if any instincts trigger, None otherwise.
        """
        action = f"Action: {tool_name} | Args: {args}"
        self._context_window.append(action)
        self._tool_sequences.append(action)
        return self._evaluate()

    def _evaluate(self) -> Optional[List[str]]:
        """Match current context window against instincts for this project."""
        if not self._current_project_id:
            return None
        context_block = "\n".join(self._context_window)
        reactions = self.instinct_manager.evaluate(self._current_project_id, context_block)
        return reactions if reactions else None

    def inject_into_response(self, result: dict, reactions: List[str]) -> dict:
        """Append instinct reactions into the result's formatted/text field."""
        injection = "\n\n--- [MemoryMesh System Instincts] ---\n" + "\n".join(reactions)

        if isinstance(result, dict):
            formatted = result.get("formatted", "")
            if formatted:
                result["formatted"] = formatted + injection
            else:
                result["instincts"] = reactions
        return result

    def get_tool_sequences(self) -> List[str]:
        """Return all recorded tool sequences for N-gram extraction."""
        return self._tool_sequences

    def reset_sequences(self):
        """Clear historical sequences (keep sliding window)."""
        self._tool_sequences.clear()
