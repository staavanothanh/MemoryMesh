"""InstinctEngine — learn patterns from memories and generate instinct rules."""

import json
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

from ..config import AppConfig
from .backend import MemoryBackend
from .instinct_store import InstinctStore

logger = logging.getLogger(__name__)

MIN_KEYWORD_LENGTH = 3
MAX_INSTINCTS_PER_USER = 50


class InstinctEngine:
    """Detect patterns in memories and generate instinct rules."""

    def __init__(self, config: AppConfig, backend: MemoryBackend, store: InstinctStore):
        self.config = config
        self.backend = backend
        self.store = store

    async def learn_from_recent(self, user_id: str):
        """Scan recent memories and generate/update instincts."""
        active_count = await self.store.count_active(user_id)
        if active_count >= MAX_INSTINCTS_PER_USER:
            logger.debug("Instinct limit reached for user %s", user_id)
            return

        recent = await self.backend.list_all(user_id, limit=100)
        if len(recent) < 5:
            return

        await self._learn_tag_patterns(user_id, recent)
        await self._learn_keyword_tags(user_id, recent)

    async def _learn_tag_patterns(self, user_id: str, memories: List[Dict[str, Any]]):
        """Learn patterns from tag co-occurrence."""
        tag_counts: Dict[str, int] = Counter()
        for m in memories:
            tags = m.get("metadata", {}).get("tags", [])
            if isinstance(tags, list):
                for t in tags:
                    tag_counts[t.lower()] += 1

        existing = await self.store.get_active_instincts(user_id)
        existing_conditions = {json.dumps(e["condition"]) for e in existing}

        for tag, count in tag_counts.items():
            if count < 3:
                continue
            condition = {"type": "tag_frequency", "tag": tag, "min_count": 3}
            if json.dumps(condition) in existing_conditions:
                continue
            confidence = min(0.9, 0.3 + count * 0.1)
            instinct_id = await self.store.add_instinct(
                user_id=user_id,
                condition=condition,
                action={"type": "suggest_tag", "tag": tag},
                confidence=round(confidence, 4),
            )
            logger.info("Learned tag instinct %s: tag '%s' (count=%d, confidence=%.2f)",
                        instinct_id, tag, count, confidence)

    async def _learn_keyword_tags(self, user_id: str, memories: List[Dict[str, Any]]):
        """Learn patterns from content keywords to tag mappings."""
        keyword_tag_map: Dict[str, Counter] = {}
        for m in memories:
            content = m.get("content", "")
            tags = m.get("metadata", {}).get("tags", [])
            if not isinstance(tags, list):
                continue
            words = self._extract_keywords(content)
            for word in words:
                if word not in keyword_tag_map:
                    keyword_tag_map[word] = Counter()
                for tag in tags:
                    keyword_tag_map[word][tag.lower()] += 1

        existing = await self.store.get_active_instincts(user_id)
        existing_conditions = {json.dumps(e["condition"]) for e in existing}

        for word, tag_counter in keyword_tag_map.items():
            total_mentions = sum(tag_counter.values())
            if total_mentions < 3:
                continue
            best_tag, best_count = tag_counter.most_common(1)[0]
            ratio = best_count / total_mentions
            if ratio < 0.6:
                continue
            condition = {"type": "keyword", "words": [word]}
            if json.dumps(condition) in existing_conditions:
                continue
            confidence = min(0.85, 0.3 + ratio * 0.5)
            instinct_id = await self.store.add_instinct(
                user_id=user_id,
                condition=condition,
                action={"type": "suggest_tag", "tag": best_tag},
                confidence=round(confidence, 4),
            )
            logger.info("Learned keyword instinct %s: '%s' -> tag '%s' (confidence=%.2f)",
                        instinct_id, word, best_tag, confidence)

    async def apply_instincts(
        self, user_id: str, content: str, tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Apply applicable instincts to a memory and return suggested modifications."""
        instincts = await self.store.get_active_instincts(user_id)
        suggestions = {"suggested_tags": [], "confidence": 0.0}
        current_tags = {t.lower() for t in (tags or [])}

        for inst in instincts:
            cond = inst["condition"]
            action = inst["action"]
            cond_type = cond.get("type")

            if cond_type == "tag_frequency":
                suggested_tag = action.get("tag", "").lower()
                if suggested_tag and suggested_tag not in current_tags:
                    suggestions["suggested_tags"].append({
                        "tag": suggested_tag,
                        "confidence": inst["confidence"],
                        "instinct_id": inst["id"],
                    })

            elif cond_type == "keyword":
                words = cond.get("words", [])
                content_lower = content.lower()
                if any(w.lower() in content_lower for w in words):
                    suggested_tag = action.get("tag", "").lower()
                    if suggested_tag and suggested_tag not in current_tags:
                        suggestions["suggested_tags"].append({
                            "tag": suggested_tag,
                            "confidence": inst["confidence"],
                            "instinct_id": inst["id"],
                        })

        if suggestions["suggested_tags"]:
            suggestions["suggested_tags"].sort(key=lambda x: x["confidence"], reverse=True)
            suggestions["confidence"] = suggestions["suggested_tags"][0]["confidence"]

        return suggestions

    async def reinforce_instinct(self, instinct_id: str, success: bool):
        """Reinforce or weaken an instinct based on whether it was applied."""
        try:
            if success:
                await self.store.update_confidence(instinct_id, 0.0, increment_trigger=True)
            else:
                await self.store.update_confidence(instinct_id, -0.05, increment_trigger=False)
        except Exception as e:
            logger.warning("Failed to reinforce instinct %s: %s", instinct_id, e)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        words = re.findall(r"[a-zA-ZÀ-ỹ]+", text.lower())
        return [w for w in words if len(w) >= MIN_KEYWORD_LENGTH and w not in _STOP_WORDS]


_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "some", "same",
    "into", "than", "that", "them", "then", "they", "this", "very", "just",
    "with", "would", "about", "their", "there", "these", "which", "when",
    "các", "có", "một", "và", "với", "cho", "được", "trong", "khi", "về",
    "như", "này", "những", "của", "làm", "là", "từ", "hoặc", "đã", "sẽ",
    "cũng", "nên", "qua", "tại", "theo", "vào", "ra", "ở", "lên", "xuống",
    "sau", "trước", "giữa", "bên", "trên", "dưới", "cùng", "nếu", "thì",
}
