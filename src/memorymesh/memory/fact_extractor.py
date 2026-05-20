"""Atomic Fact Extractor — extract standalone facts from conversation via LLM."""

import json
import logging
from typing import List, Optional, Dict, Any

from ..config import AppConfig
from ..router import RouterClient
from ..prompts import ATOMIC_FACT_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


class FactExtractor:
    def __init__(self, config: AppConfig, router: RouterClient):
        self.config = config
        self.router = router

    async def extract_facts(self, conversation: str) -> List[Dict[str, Any]]:
        """Extract atomic facts from a conversation via LLM.

        Returns a list of fact dicts with keys: fact, confidence, tags.
        """
        if not conversation.strip():
            return []
        prompt = ATOMIC_FACT_EXTRACT_PROMPT.format(conversation=conversation)
        try:
            response = await self.router.call_llm(prompt)
            data = json.loads(response)
            raw_facts = data.get("facts", [])
            if not isinstance(raw_facts, list):
                logger.warning("FactExtractor: 'facts' is not a list: %s", type(raw_facts))
                return []
            return self._deduplicate_and_validate(raw_facts)
        except Exception as e:
            logger.error("Fact extraction failed: %s", e)
            return []

    def _deduplicate_and_validate(
        self, raw_facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen = set()
        valid = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_text = item.get("fact", "")
            if not isinstance(fact_text, str) or not fact_text.strip():
                continue
            norm = fact_text.strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            valid.append({
                "fact": fact_text.strip(),
                "confidence": item.get("confidence", "medium") if isinstance(item.get("confidence"), str) else "medium",
                "tags": item.get("tags", []) if isinstance(item.get("tags"), list) else [],
            })
        logger.info("FactExtractor: %d raw -> %d unique valid facts", len(raw_facts), len(valid))
        return valid

    def _confidence_to_importance(self, confidence: str) -> int:
        mapping = {"high": 4, "medium": 3, "low": 2}
        return mapping.get(confidence.lower(), 3)
