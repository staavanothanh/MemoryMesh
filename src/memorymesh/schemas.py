from typing import TypedDict, List

class MemoryRecord(TypedDict):
    id: str
    user_id: str
    content: str
    tags: List[str]
    importance: int
    timestamp: str

class SearchResult(TypedDict):
    id: str
    content: str
    score: float
    tags: List[str]
    importance: int
    timestamp: str

class ToolOutput(TypedDict):
    status: str  # "success" or "error"
    data: dict | list | str | None
    error: str | None