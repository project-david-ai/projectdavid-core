import os
import time
from typing import List, Optional

import jwt

# 1. Load from Environment Variables
SECRET_KEY = os.getenv("SANDBOX_AUTH_SECRET", "dev_secret_key_change_this")
ALGORITHM = "HS256"


def mint_computer_ticket(
    user_id: str, room_id: str, scopes: Optional[List[str]] = None
) -> str:
    """
    SERVER SIDE ONLY.
    Creates a temporary, short-lived JWT for WebSocket connection.

    Args:
        user_id: The ID of the user (observer).
        room_id: The ID of the thread/room to join.
        scopes: List of permissions. Defaults to ["observation"] for safety.
    """

    # 2. Default Scope Logic (Safety First)
    # If no scopes are provided, assume read-only observation.
    if scopes is None:
        scopes = ["observation"]

    # 3. Define Claims (Aligned with Agent Tokens)
    payload = {
        "sub": user_id,  # Subject (User)
        "room": room_id,  # Context (Room) - Critical for your security check
        "scopes": scopes,  # Capabilities - Aligned with agent structure
        "type": "ws_ticket",  # Purpose - distinct from login tokens
        "exp": time.time() + 30,  # Expiration - 30s window to connect
        "iat": time.time(),  # Issued At
    }

    # 4. Sign the Token
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return token
