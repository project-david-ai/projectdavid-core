"""Server-to-server sandbox admission and owner-scoped artifact upload."""

import hashlib
import mimetypes
import os
import stat
import tempfile

import httpx


def _connection():
    key = os.getenv("ADMIN_API_KEY")
    if not key:
        raise RuntimeError("Sandbox service credentials unavailable.")
    base = os.getenv("SANDBOX_API_BASE_URL", "http://fastapi_cosmic_catalyst:9000")
    return httpx.AsyncClient(
        base_url=base.rstrip("/"),
        headers={"X-API-Key": key},
        timeout=httpx.Timeout(60, connect=5),
        follow_redirects=False,
        trust_env=False,
    )


def create_workspace(base_dir: str, owner_id: str) -> str:
    """Keep new execution files under an account-specific, non-user-controlled path."""
    if not owner_id:
        raise RuntimeError("Sandbox workspace owner is required.")
    owner_hash = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    account_dir = os.path.join(base_dir, owner_hash)
    os.makedirs(account_dir, exist_ok=True)
    return tempfile.mkdtemp(prefix="execution-", dir=account_dir)


async def admit_session(token: str, *, mode: str, room=None, role="executor") -> str:
    async with _connection() as client:
        response = await client.post(
            "/v1/internal/sandbox/admission",
            json={"token": token, "mode": mode, "room": room, "role": role},
        )
        response.raise_for_status()
        owner = response.json()["owner_id"]
        if not isinstance(owner, str) or not owner:
            raise RuntimeError("Sandbox admission did not return an owner.")
        return owner


async def upload_artifact(owner_id: str, filename: str, path: str) -> dict:
    if not owner_id:
        raise RuntimeError("Sandbox artifact owner is required.")
    # Avoid following an artifact symlink outside the execution directory.
    fd = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Sandbox artifact is not a regular file.")
        async with _connection() as client:
            response = await client.post(
                "/v1/internal/sandbox/artifacts",
                data={"owner_id": owner_id},
                files={
                    "file": (
                        filename,
                        stream,
                        mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    )
                },
            )
            response.raise_for_status()
            return response.json()


async def verify_pending_erasure(owner_id: str) -> None:
    async with _connection() as client:
        response = await client.post(
            "/v1/internal/sandbox/erasure-pending", json={"owner_id": owner_id}
        )
        response.raise_for_status()
        ack = response.json()
        if (
            not isinstance(ack, dict)
            or ack.get("owner_id") != owner_id
            or ack.get("pending") is not True
        ):
            raise RuntimeError("Pending erasure not confirmed.")
