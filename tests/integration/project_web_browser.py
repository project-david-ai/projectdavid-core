import logging
import os
import shutil
import subprocess
from typing import List

import html2text
import requests
from playwright.async_api import async_playwright

# Import your cache class
from src.api.entities_api.cache.web_cache import WebSessionCache

logger = logging.getLogger("UniversalWebReader")


class UniversalWebReader:
    """
    The 'Eyes' of the agent.
    Responsibility: Fetch raw content (Static or Dynamic), clean it, chunk it,
    and hand it off to the WebSessionCache.
    """

    def __init__(self, cache_service: WebSessionCache):
        self.cache = cache_service

        # HTML to Markdown Configuration
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = False
        self.converter.ignore_images = True
        self.converter.ignore_tables = False
        self.converter.body_width = 0

        # Config
        self.CHUNK_SIZE = 4000
        self.browser_ws = os.getenv(
            "BROWSER_WS_ENDPOINT", None
        )  # e.g. ws://browser:3000
        self.has_curl = shutil.which("curl") is not None

        self.garbage_triggers = [
            "enable javascript",
            "please wait",
            "loading...",
            "javascript is required",
            "checking your browser",
            "turn on js",
        ]

    async def read(self, url: str, force_refresh: bool = False) -> str:
        """
        Action: Scrape a URL.
        1. Checks Redis (via WebSessionCache).
        2. If missing, Fetches (Curl -> Browserless).
        3. Saves to Redis.
        4. Returns Page 0.
        """
        # 1. Check Cache
        if not force_refresh:
            # We check if the session exists in your Redis Cache
            cached_session = await self.cache.get_session(url)
            if cached_session:
                logger.info(f"⚡ Cache Hit for {url}")
                # Return Page 0 using your cache's formatting logic
                return await self.cache.get_page_view(url, 0)

        # 2. Fetch Content
        logger.info(f"🌐 Fetching fresh content: {url}")
        content = self._fetch_static(url)
        source = "Static (Fast)"

        # 3. Validate & Failover to Browser
        if not self._is_valid_content(content):
            logger.info(
                "⚠️ Static fetch failed/garbage. Switching to Sidecar Browser..."
            )
            content = await self._fetch_dynamic(url)
            source = "Dynamic (Browser)"

        # 4. Process & Chunk
        clean_text = content.strip()
        if not clean_text:
            return "❌ Error: Could not extract any text from this URL."

        chunks = self._chunk_text(clean_text)

        # 5. Save to your Redis Cache
        await self.cache.save_session(
            url=url, full_text=clean_text, chunks=chunks, source=source
        )

        # 6. Return Page 0
        return await self.cache.get_page_view(url, 0)

    async def scroll(self, url: str, page: int) -> str:
        """
        Action: Scroll.
        Direct pass-through to the cache logic.
        """
        return await self.cache.get_page_view(url, page)

    # --- INTERNAL HELPERS ---

    def _chunk_text(self, text: str) -> List[str]:
        """Splits text into 4000-char pages."""
        return [
            text[i : i + self.CHUNK_SIZE] for i in range(0, len(text), self.CHUNK_SIZE)
        ]

    def _is_valid_content(self, text: str) -> bool:
        """Detects if we got a 'Please enable JS' stub."""
        if not text or len(text) < 300:
            return False
        header = text.lower()[:500]
        return not any(trigger in header for trigger in self.garbage_triggers)

    # --- NETWORK LAYERS (Docker Optimized) ---

    def _fetch_static(self, url: str) -> str:
        """Layer 1: Curl (Fastest, avoids some bot blocks)."""
        if self.has_curl:
            try:
                # -L follows redirects, -s silent, --max-time prevents hangs
                cmd = [
                    "curl",
                    "-L",
                    "-s",
                    "--max-time",
                    "10",
                    "--user-agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    url,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0 and len(result.stdout) > 100:
                    return self.converter.handle(result.stdout)
            except Exception as e:
                logger.warning(f"Curl error: {e}")

        # Fallback: Requests
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return self.converter.handle(resp.text)
        except Exception:
            return ""

    async def _fetch_dynamic(self, url: str) -> str:
        """Layer 2: Browserless Sidecar (Heavy duty)."""
        async with async_playwright() as p:
            browser = None
            try:
                if self.browser_ws:
                    # Connect to the 'browser' container defined in docker-compose
                    browser = await p.chromium.connect_over_cdp(self.browser_ws)
                else:
                    # Local fallback (dev mode)
                    browser = await p.chromium.launch(headless=True)

                context = await browser.new_context()
                page = await context.new_page()

                # Extended timeout for heavy sites
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")

                # Optional: Smart wait for text hydration
                try:
                    await page.wait_for_selector("body", timeout=5000)
                except Exception:
                    pass

                html = await page.content()
                return self.converter.handle(html)
            except Exception as e:
                logger.error(f"Browser error: {e}")
                return f"Error reading page via browser: {e}"
            finally:
                if browser:
                    await browser.close()
