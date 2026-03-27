# src/api/entities_api/cache/web_cache.py
import asyncio
import hashlib
import json
import os
from typing import Dict, Optional, Union

from fastapi import Depends
from redis import Redis as SyncRedis

try:
    import redis.asyncio as redis_async_lib  # Renamed for clarity in get_redis
    from redis.asyncio import Redis as AsyncRedis
except ImportError:

    class AsyncRedis:
        pass

    redis_async_lib = None


# Web sessions are ephemeral. 1 hour (3600s) is usually enough for an agent to read a doc.
REDIS_WEB_TTL = int(os.getenv("REDIS_WEB_TTL_SECONDS", "3600"))
# ✅ ADD: Default Redis URL from your docker-compose environment
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


class WebSessionCache:
    """
    Manages the storage of scraped web content.
    Splits content into pages (chunks) to support the 'Scroll' capability.
    Also supports 'Search' to scan chunks without loading them into the LLM context.
    """

    def __init__(self, redis: Union[SyncRedis, "AsyncRedis"]):
        self.redis = redis

    def _cache_key(self, url: str) -> str:
        """
        We hash the URL to create a safe, consistent Redis key.
        Prefix: 'web_session:'
        """
        url_hash = hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"web_session:{url_hash}"

    async def get_session(self, url: str) -> Optional[Dict]:
        """Retrieves the full session data (all chunks) for a URL."""
        key = self._cache_key(url)

        if isinstance(self.redis, AsyncRedis):
            raw = await self.redis.get(key)
        else:
            raw = await asyncio.to_thread(self.redis.get, key)

        return json.loads(raw) if raw else None

    async def save_session(self, url: str, full_text: str, chunks: list, source: str):
        """
        Saves a new web scrape.
        """
        key = self._cache_key(url)

        payload = {
            "url": url,
            "source": source,
            "total_pages": len(chunks),
            "chunks": chunks,
            "full_length": len(full_text),
            "timestamp": asyncio.get_event_loop().time(),
        }

        data = json.dumps(payload)

        if isinstance(self.redis, AsyncRedis):
            await self.redis.set(key, data, ex=REDIS_WEB_TTL)
        else:
            await asyncio.to_thread(self.redis.set, key, data, ex=REDIS_WEB_TTL)

    async def get_page_view(self, url: str, page_index: int) -> str:
        """
        Logic to retrieve a specific slice (page) for the LLM.
        Returns the formatted string ready for the prompt.
        """
        data = await self.get_session(url)

        if not data:
            return "❌ Error: Cache miss. This URL has not been loaded yet. Please call `web_read` first."

        total_pages = data["total_pages"]

        if page_index >= total_pages:
            return f"❌ Error: Page {page_index} out of bounds. Total pages: {total_pages}. (Try page {total_pages - 1})"

        chunk_content = data["chunks"][page_index]

        # --- Reconstruct the Agent-Friendly View ---
        output = []
        output.append(f"--- 🌐 WEB BROWSER: {data['url']} ---")
        output.append(
            f"--- 📊 SOURCE: {data['source']} | VIEW: Page {page_index} of {total_pages - 1} ---"
        )
        output.append(f"--- 📏 SIZE: {data['full_length']} chars total ---\n")
        output.append(chunk_content)
        output.append("\n" + "=" * 30)

        if page_index < total_pages - 1:
            output.append(
                f"👉 SYSTEM NOTICE: Content continues. To read the next part, call `web_scroll(url='{data['url']}', page={page_index + 1})`"
            )
        else:
            output.append("✅ SYSTEM NOTICE: End of Document.")

        return "\n".join(output)

    async def search_session(self, url: str, query: str) -> str:
        """
        Scans ALL cached chunks for a specific keyword/phrase.
        Returns extracted context windows around the match, not the whole page.
        This prevents 'Context Pollution' by filtering the noise on the server side.
        """
        data = await self.get_session(url)

        if not data:
            return "❌ Error: Cache miss. This URL has not been loaded yet. Please call `web_read` first."

        chunks = data["chunks"]
        query_lower = query.lower()
        results = []

        # Iterate through all chunks to find matches
        for index, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            if query_lower in chunk_lower:
                # Extract a snippet (150 chars before and after the match)
                start_idx = chunk_lower.find(query_lower)
                start_context = max(0, start_idx - 150)
                end_context = min(len(chunk), start_idx + len(query) + 150)

                snippet = chunk[start_context:end_context].replace("\n", " ")
                results.append(f"--- 📄 FOUND IN PAGE {index} ---\n...{snippet}...")

                # Safety Limit: Stop if we find too many matches (e.g., searching for "the")
                if len(results) >= 15:
                    break

        if not results:
            return f"❌ Keyword '{query}' not found in any of the {len(chunks)} pages."

        # --- Format the Search Result ---
        output = []
        output.append(f"--- 🔍 SEARCH RESULTS: '{query}' in {data['url']} ---")
        output.extend(results)

        if len(results) >= 15:
            output.append("\n(⚠️ Search limit reached. There may be more matches.)")

        output.append("\n" + "=" * 30)
        output.append(
            "👉 SYSTEM HINT: Use `web_scroll(url, page=X)` to read the full context of a specific page."
        )

        return "\n".join(output)
