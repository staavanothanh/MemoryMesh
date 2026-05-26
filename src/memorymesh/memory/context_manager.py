"""Context Manager — stateless cursor-based pagination with dynamic scoring.

Implements keyset (cursor) pagination to avoid LIMIT/OFFSET performance
degradation on deep pages. Scoring is pushed down to SQLite via CTE.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..utils.tokenization import estimate_tokens, count_tokens_exact

logger = logging.getLogger(__name__)

CHUNK_SIZE = 50


@dataclass
class Cursor:
    last_score: float = 9999.0
    last_id: str = ""
    page: int = 1

    def to_dict(self) -> dict:
        return {"last_score": self.last_score, "last_id": self.last_id, "page": self.page}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Cursor":
        if not d:
            return Cursor()
        return Cursor(
            last_score=d.get("last_score", 9999.0),
            last_id=d.get("last_id", ""),
            page=d.get("page", 1),
        )


def _build_dynamic_score_query(
    user_id: str,
    cursor: Cursor,
    chunk_size: int = CHUNK_SIZE,
    level_weights: Optional[Dict[str, float]] = None,
) -> tuple:
    """Build SQL with keyset pagination and dynamic scoring pushed to DB via CTE."""
    w_session = level_weights.get("session", 2.0) if level_weights else 2.0
    w_user = level_weights.get("user", 1.5) if level_weights else 1.5
    w_knowledge = level_weights.get("knowledge", 1.0) if level_weights else 1.0
    decay_hours = 24.0

    sql = f"""
        WITH scored AS (
            SELECT id, content, metadata_json, importance, level, created_at,
                   (importance *
                       CASE level
                           WHEN 'session' THEN {w_session}
                           WHEN 'user' THEN {w_user}
                           ELSE {w_knowledge}
                       END *
                       exp(-(julianday('now') - julianday(substr(created_at, 1, 19))) * 24.0 / {decay_hours})
                   ) AS dynamic_score
            FROM memories
            WHERE user_id = ? AND deleted = 0
        )
        SELECT * FROM scored
        WHERE (dynamic_score < ?) OR (dynamic_score = ? AND id < ?)
        ORDER BY dynamic_score DESC, id DESC
        LIMIT ?
    """
    params = [user_id, cursor.last_score, cursor.last_score, cursor.last_id, chunk_size]
    return sql, params


class ContextManager:
    """Manages stateless pagination of memories with token budget control."""

    def __init__(self, backend):
        self._backend = backend

    async def get_context_page(
        self,
        user_id: str,
        max_tokens: int = 1000,
        cursor: Optional[dict] = None,
        level_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Fetch one page of context, respecting token budget via keyset cursor.

        Uses CTE for dynamic scoring + keyset pagination. Token budget is
        estimated fast first, then confirmed exact via thread-pool tiktoken
        only when close to the limit.
        """
        cur = Cursor.from_dict(cursor)
        sql, params = _build_dynamic_score_query(user_id, cur, level_weights=level_weights)
        try:
            db = getattr(self._backend, "_db", None)
            if not db:
                return {"results": [], "next_cursor": None, "tokens_used": 0, "has_more": False}

            rows = []
            async with db.execute(sql, params) as db_cursor:
                async for row in db_cursor:
                    rows.append(row)
        except Exception as e:
            logger.error("Context page query failed: %s", e)
            return {"results": [], "next_cursor": None, "tokens_used": 0, "has_more": False}

        if not rows:
            return {"results": [], "next_cursor": None, "tokens_used": 0, "has_more": False}

        selected = []
        total_tokens = 0
        last_row = None

        for row in rows:
            content = row["content"] if isinstance(row, dict) else row[1]
            meta_str = row.get("metadata_json", "{}") if isinstance(row, dict) else (row[2] if len(row) > 2 else "{}")
            mem_id = row.get("id", "") if isinstance(row, dict) else (row[0] if len(row) > 0 else "")
            importance = row.get("importance", 3) if isinstance(row, dict) else (row[3] if len(row) > 3 else 3)
            level = row.get("level", "user") if isinstance(row, dict) else (row[4] if len(row) > 4 else "user")
            created_at = row.get("created_at", "") if isinstance(row, dict) else (row[5] if len(row) > 5 else "")
            raw_score = row.get("dynamic_score", 0.0) if isinstance(row, dict) else (row[6] if len(row) > 6 else 0.0)

            est = estimate_tokens(content) + estimate_tokens(meta_str)
            if total_tokens + est > max_tokens:
                exact = await count_tokens_exact(content + meta_str)
                if total_tokens + exact > max_tokens:
                    break
                total_tokens += exact
            else:
                total_tokens += est

            selected.append({
                "id": mem_id,
                "content": content,
                "metadata": json.loads(meta_str) if isinstance(meta_str, str) else meta_str,
                "importance": importance,
                "level": level,
                "score": float(raw_score),
            })
            last_row = row

        has_more = len(selected) < len(rows) or len(rows) >= CHUNK_SIZE

        next_cursor = None
        if has_more and last_row:
            ls = last_row.get("dynamic_score", 1.0) if isinstance(last_row, dict) else (last_row[6] if len(last_row) > 6 else 1.0)
            lid = last_row.get("id", "") if isinstance(last_row, dict) else (last_row[0] if len(last_row) > 0 else "")
            next_cursor = Cursor(
                last_score=float(ls),
                last_id=lid,
                page=cur.page + 1,
            ).to_dict()

        return {
            "results": selected,
            "next_cursor": next_cursor,
            "tokens_used": total_tokens,
            "has_more": has_more,
        }
