"""A small async client for the memory judge (any OpenAI-compatible chat endpoint)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChatClient:
    """Thin async wrapper over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 2,
        max_concurrency: int = 64,
    ):
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "") or None
        self.api_key = os.getenv(api_key_env, "") or "EMPTY"
        self.timeout = timeout
        self.max_retries = max_retries
        self._max_concurrency = max_concurrency
        self._client = None
        self._sem: Optional[asyncio.Semaphore] = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _ensure_sem(self) -> asyncio.Semaphore:
        # Created lazily so it binds to the loop that actually runs the rollout.
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrency)
        return self._sem

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Return the assistant text, or ``None`` if every attempt failed."""
        client = self._ensure_client()
        sem = self._ensure_sem()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                async with sem:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
                last_error = ValueError("empty completion")
            except Exception as exc:  # noqa: BLE001 - any API failure is recoverable here
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))
        logger.warning("[simmem] chat completion failed after retries: %s", last_error)
        return None
