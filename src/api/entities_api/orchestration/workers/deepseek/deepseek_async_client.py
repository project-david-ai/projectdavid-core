import asyncio
import json
from typing import AsyncGenerator, Dict, List, Union

import httpx


class AsyncDeepSeekClient:
    """
    Lightweight streaming wrapper around the DeepSeek Chat Completion API.
    Designed to be symmetric with `AsyncHyperbolicClient`.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        max_retries: int = 3,
        timeout: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(
            timeout=timeout, headers=self.headers, follow_redirects=True, http2=True
        )

    @staticmethod
    def _normalise_messages(
        prompt_or_messages: Union[str, List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    async def stream_chat_completion(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Yield content tokens from DeepSeek in streaming mode.

        Usage
        -----
        async for token in client.stream_chat_completion("Hello DeepSeek!"):
            print(token, end="", flush=True)
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": self._normalise_messages(prompt_or_messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for raw in resp.aiter_lines():
                        if not raw:
                            continue
                        if raw.startswith("data: "):
                            raw = raw[6:]
                        if raw.strip() == "[DONE]":
                            return
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            print(f"[!] malformed JSON: {raw[:100]} …")
                            continue
                        if chunk.get("object") == "error":
                            msg = chunk.get("message", "Unknown DeepSeek error")
                            raise RuntimeError(msg)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    return
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise
                backoff = 2 ** (attempt - 1)
                print(f"[Retry {attempt}] {exc}  – sleeping {backoff}s")
                await asyncio.sleep(backoff)

    async def aclose(self):
        await self.client.aclose()
