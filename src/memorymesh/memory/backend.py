from typing import Protocol, List, Dict, Any, Optional

class MemoryBackend(Protocol):
    """Abstract interface for memory storage."""

    async def add(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "user",
    ) -> str:
        """Store a memory and return its ID."""
        ...

    async def search(
        self,
        embedding: List[float],
        user_id: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
        level_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find memories closest to the embedding for a user."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        ...

    async def update_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for an existing memory."""
        ...

    async def update(
        self, memory_id: str, content: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update content and metadata of an existing memory."""
        ...

    async def get_with_embeddings(
        self, user_id: str, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get memories with their embeddings for processing (e.g. consolidation)."""
        ...

    async def get_with_embeddings_by_ids(
        self, ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get specific memories by IDs, with embeddings (for rollback backup)."""
        ...

    async def list_all(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all memories of a user, with pagination."""
        ...

    async def list_by_tag(self, user_id: str, tag: str) -> List[Dict[str, Any]]:
        """List non-deleted memories matching a specific tag."""
        ...

    async def delete_by_tag(self, user_id: str, tag: str) -> int:
        """Delete all non-deleted memories matching a tag. Returns count deleted."""
        ...