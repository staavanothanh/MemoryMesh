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
        self._semaphore = asyncio.Semaphore(3)
        self.config = config
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._client = httpx.AsyncClient(timeout=config.timeout_s)

    async def close(self):
        await self._client.aclose()

    async def call_llm(self, prompt: str, model: Optional[str] = None) -> str:
        """Call LLM via 9Router with retry and fallback."""
        async with self._semaphore:
            return await self._call_llm_impl(prompt, model)

    async def call_llm_background(self, prompt: str, *, json_mode: bool = False) -> str:
        """Call LLM using free model pool cascade for background tasks.

        Tries each model in background_model_pool in sequence.
        All calls use temperature=0.0 for deterministic output.
        If json_mode=True, also sends response_format={"type": "json_object"}.
        If all free models fail, falls back to normal call_llm (paid).
        """
        pool = self.config.background_model_pool
        if not pool:
            return await self.call_llm(prompt)

        kwargs = {"temperature": 0.0}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for model in pool:
            model = model.strip()
            if not model:
                continue
            try:
                async with self._semaphore:
                    return await self._call(model, prompt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning("Background model %s failed: %s", model, e)
                continue

        logger.warning("All background models failed, falling back to call_llm: %s", last_error)
        return await self.call_llm(prompt)

    async def _call_llm_impl(self, prompt: str, model: Optional[str] = None) -> str:
        if self._failure_count >= 3:
            raise LLMUnavailableError("Circuit breaker open: too many failures")

        primary = model or self.config.default_model
        fallback = self.config.fallback_model
        models_to_try = [primary] if primary == fallback else [primary, fallback]

        last_error = None
        for m in models_to_try:
            for attempt in range(self.config.max_retries):
                try:
                    return await self._call(m, prompt)
                except RouterError as e:
                    last_error = e
                    logger.warning("LLM call failed (model=%s, attempt %d/%d): %s", m, attempt+1, self.config.max_retries, e)
                    await asyncio.sleep(2 ** attempt)

        self._failure_count += 1
        if self._failure_count >= 3:
            raise LLMUnavailableError("Circuit breaker open: too many failures")
        raise last_error or RouterError(primary, self.config.max_retries, "Unknown error")

    async def _call(self, model: str, prompt: str, temperature: float = None, response_format: dict = None) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = response_format
        try:
            response = await self._client.post(
                f"{self.config.url}/chat/completions",
                json=body,
            )
            if response.status_code == 200:
                self._failure_count = 0
                raw_text = response.text
                logger.debug("LLM response (first 200 chars): %s", raw_text[:200])
                try:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
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