"""PostToolUse hooks for MemoryMesh — event-driven extensions."""

import logging
from typing import Any, Dict, List, Callable, Awaitable

logger = logging.getLogger(__name__)

HookFunc = Callable[..., Awaitable[None]]


class HookRegistry:
    """Lightweight pub-sub event registry."""

    def __init__(self):
        self._hooks: Dict[str, List[HookFunc]] = {}

    def register(self, event: str, hook: HookFunc):
        self._hooks.setdefault(event, []).append(hook)
        logger.debug("Hook registered for event '%s'", event)

    async def trigger(self, event: str, **kwargs: Any):
        for hook in self._hooks.get(event, []):
            try:
                await hook(**kwargs)
            except Exception as e:
                logger.error("Hook '%s' failed: %s", event, e)


hooks = HookRegistry()
