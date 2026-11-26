"""Require a positive account-specific sandbox drain acknowledgement."""

import os

import httpx
from fastapi import HTTPException


def require_sandbox_drain(user_id: str) -> None:
    key = os.getenv("ADMIN_API_KEY")
    url = os.getenv("SANDBOX_SERVER_URL")
    if not key or not url:
        raise HTTPException(503, "Sandbox drain configuration unavailable.")
    try:
        with httpx.Client(
            timeout=httpx.Timeout(30, connect=5),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.post(
                url.rstrip("/") + "/ws/internal/account-drain",
                headers={"X-API-Key": key},
                json={"owner_id": user_id},
            )
            if response.status_code == 409:
                raise HTTPException(
                    409, "Account deletion is pending sandbox reconciliation."
                )
            response.raise_for_status()
            ack = response.json()
            if (
                not isinstance(ack, dict)
                or ack.get("owner_id") != user_id
                or ack.get("drained") is not True
            ):
                raise ValueError("Invalid sandbox drain acknowledgement.")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Sandbox drain could not be confirmed.") from None
