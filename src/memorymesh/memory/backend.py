from typing import Protocol, List, Dict, Any, Optional

class MemoryBackend(Protocol):
    """Abstract interface for memory storage."""

    async def add(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory and return its ID."""
        ...

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find memories closest to the embedding for a user."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        ...

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all memories of a user, with pagination."""
        ...