"""Optional, bounded retrieval of public website icons; never fetch page content."""

import asyncio
import base64
import io
import os
import re
from typing import Optional
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from src.api.entities_api.services.outbound_network_policy import (
    public_address,
    resolve_addresses,
)

MAX_DOWNLOAD = 256 * 1024
MAX_DIMENSION = 512
ICON_SIZE = (32, 32)
DEADLINE_SECONDS = 2.0
DATA_PREFIX = "data:image/png;base64,"
_HOST = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?",
    re.ASCII,
)
_FORMATS = ("PNG", "ICO", "JPEG", "GIF", "WEBP")
_EXTENSIONS = (".ico", ".png", ".jpg", ".jpeg", ".gif", ".webp")


def favicon_enabled() -> bool:
    return os.getenv("WEB_FAVICONS_ENABLED", "false").strip().lower() in {"1", "true"}


def _safe_url(value: str) -> Optional[httpx.URL]:
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    if any(ord(c) <= 32 or ord(c) == 127 or c == "\\" for c in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or not _HOST.fullmatch(host)
            or len(host) > 253
            or parsed.netloc.lower() not in (host, f"{host}:443")
        ):
            return None
        url = httpx.URL(value)
        if url.host != host or url.scheme != "https" or url.port not in (None, 443):
            return None
        return url
    except (ValueError, httpx.InvalidURL):
        return None


def favicon_origin(page_url: str) -> Optional[str]:
    url = _safe_url(page_url)
    return f"https://{url.host}" if url is not None else None


def favicon_candidate(
    page_url: str, declared_icon: Optional[str] = None
) -> Optional[httpx.URL]:
    origin = favicon_origin(page_url)
    if origin is None:
        return None
    if isinstance(declared_icon, str) and 0 < len(declared_icon) <= 2048:
        # Do not allow URL parser normalization to hide unsafe raw characters.
        if not any(ord(c) <= 32 or ord(c) == 127 or c == "\\" for c in declared_icon):
            try:
                url = _safe_url(urljoin(page_url, declared_icon))
                if (
                    url is not None
                    and favicon_origin(str(url)) == origin
                    and not url.query
                    and not url.fragment
                    and url.path.lower().endswith(_EXTENSIONS)
                ):
                    return url
            except ValueError:
                pass
    return httpx.URL(f"{origin}/favicon.ico")


def _public_address(value: str) -> bool:
    return public_address(value)


async def _resolve_public(host: str) -> Optional[str]:
    addresses = await resolve_addresses(host, 443)
    if not addresses or not all(_public_address(address) for address in addresses):
        return None
    # Pin one verified address; never let the HTTP layer resolve the hostname again.
    return next((address for address in addresses if ":" not in address), addresses[0])


def _normalize_image(body: bytes) -> Optional[str]:
    if not body or len(body) > MAX_DOWNLOAD:
        return None
    try:
        with Image.open(io.BytesIO(body), formats=list(_FORMATS)) as image:
            if not (
                0 < image.width <= MAX_DIMENSION and 0 < image.height <= MAX_DIMENSION
            ):
                return None
            image.seek(0)
            image = image.convert("RGBA")
            image.thumbnail(ICON_SIZE, Image.Resampling.LANCZOS)
            # New canvas strips source metadata, embedded profiles and animation.
            canvas = Image.new("RGBA", ICON_SIZE)
            canvas.paste(image, ((32 - image.width) // 2, (32 - image.height) // 2))
            output = io.BytesIO()
            canvas.save(output, format="PNG")
            return DATA_PREFIX + base64.b64encode(output.getvalue()).decode("ascii")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


async def _fetch_icon(url: httpx.URL) -> Optional[str]:
    address = await _resolve_public(url.host)
    if address is None:
        return None
    # The numeric URL pins the TCP destination. httpcore's sni_hostname request
    # extension preserves the original hostname for TLS SNI AND verification.
    # A fresh client prevents pooling across distinct hostnames sharing an IP.
    pinned = url.copy_with(host=address)
    async with httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        verify=True,
        timeout=httpx.Timeout(DEADLINE_SECONDS),
    ) as client:
        async with client.stream(
            "GET",
            pinned,
            headers={
                "Host": url.host,
                "Accept": "image/*",
                "Accept-Encoding": "identity",
            },
            extensions={"sni_hostname": url.host},
        ) as response:
            if response.status_code != 200:
                return None
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                return None
            content_length = response.headers.get("content-length")
            if content_length is not None and not (
                0 < int(content_length) <= MAX_DOWNLOAD
            ):
                return None
            data = bytearray()
            async for chunk in response.aiter_raw(chunk_size=8192):
                if len(data) + len(chunk) > MAX_DOWNLOAD:
                    return None
                data.extend(chunk)
    return _normalize_image(bytes(data))


async def resolve_favicon(
    page_url: str, declared_icon: Optional[str] = None
) -> Optional[str]:
    """Return inert PNG data, or None. Failure must never fail a web tool."""
    if not favicon_enabled():
        return None
    try:
        url = favicon_candidate(page_url, declared_icon)
        if url is None:
            return None
        return await asyncio.wait_for(_fetch_icon(url), timeout=DEADLINE_SECONDS)
    except Exception:
        # No URLs or response bodies in logs; callers use the globe fallback.
        return None
