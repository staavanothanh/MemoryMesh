"""Async client for 9Router with retry, fallback, and circuit breaker."""

import asyncio
import logging
from typing import Optional
import json
import httpx

from .config import RouterConfig
from .errors import RouterError, LLMUnavailableError

logger = logging.getLogger(__name__)


class RouterClient:
    def __init__(self, config: RouterConfig):
        self.config = config
        self._failure_count = 0
        self._last_failure_time = 0.0

    async def call_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Call LLM via 9Router with retry and fallback."""
        model = model or self.config.default_model
        models_to_try = [model]
        if model != self.config.fallback_model:
            models_to_try.append(self.config.fallback_model)

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await self._call(model, prompt)
            except RouterError as e:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt+1, self.config.max_retries, e)
                last_error = e
                if attempt == self.config.max_retries - 1:
                    # Try fallback model on final retry
                    if len(models_to_try) > 1:
                        model = self.config.fallback_model
                        try:
                            return await self._call(model, prompt)
                        except RouterError as e2:
                            last_error = e2
                await asyncio.sleep(2 ** attempt)

        # Circuit breaker: track consecutive failures
        self._failure_count += 1
        if self._failure_count >= 3:
            raise LLMUnavailableError("Circuit breaker open: too many failures")
        raise last_error or RouterError(model, self.config.max_retries, "Unknown error")

    async def _call(self, model: str, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                response = await client.post(
                    f"{self.config.url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if response.status_code == 200:
                    self._failure_count = 0
                    raw_text = response.text
                    logger.debug("LLM response (first 200 chars): %s", raw_text[:200])
                    try:
                        # Thử parse bình thường trước
                        data = response.json()
                        return data["choices"][0]["message"]["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Nếu lỗi, thử parse object JSON đầu tiên trong chuỗi
                        decoder = json.JSONDecoder()
                        try:
                            data, _ = decoder.raw_decode(raw_text)
                            return data["choices"][0]["message"]["content"]
                        except (json.JSONDecodeError, KeyError, IndexError) as e2:
                            raise RouterError(
                                model, 0,
                                Exception(f"JSON parse error: {e2}. Raw (500 chars): {raw_text[:500]}")
                            )
                else:
                    raise RouterError(
                        model, 0, Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                    )
        except httpx.TimeoutException as e:
            raise RouterError(model, 0, e)
        except httpx.ConnectError as e:
            raise RouterError(model, 0, e)