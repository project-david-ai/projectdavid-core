import os
import time

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.entities_api.dependencies import get_api_key
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.utilities.mint_computer_ticket import mint_computer_ticket

router = APIRouter()
logging_utility = LoggingUtility()

# Configuration
SECRET_KEY = os.getenv("SANDBOX_AUTH_SECRET")
SANDBOX_URL = os.getenv("SHELL_SERVER_EXTERNAL_URL", "ws://localhost:8000/ws/computer")
ALGORITHM = "HS256"
TOKEN_EXPIRATION_SECONDS = 60  # Short life! Client must connect immediately.


# --- Response Model ---
class SandboxAccessResponse(BaseModel):
    token: str
    sandbox_url: str
    expires_in: int


@router.post("/tools/sandbox/authorize", response_model=SandboxAccessResponse)
async def authorize_sandbox_execution(
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Generates a short-lived JWT for the Sandbox WebSocket.

    1. Validates the User's API Key (via Depends).
    2. Signs a JWT with the User ID.
    3. Returns the JWT + Sandbox URL.
    """

    if not SECRET_KEY:
        logging_utility.error("SANDBOX_AUTH_SECRET is missing in Main API env")
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    # 1. Create the Payload
    # We embed the User ID so the Sandbox knows who is running code
    payload = {
        "sub": str(auth_key.user_id),
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRATION_SECONDS,
        "type": "sandbox_access",
    }

    # 2. Sign the Token
    try:
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logging_utility.error(f"Failed to sign sandbox token: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not generate credentials")

    logging_utility.info(f"Generated Sandbox Token for User ID: {auth_key.user_id}")

    return {
        "token": token,
        "sandbox_url": SANDBOX_URL,
        "expires_in": TOKEN_EXPIRATION_SECONDS,
    }


@router.get("/computer/session")
def get_sandbox_connection_ticket(
    room_id: str = Query(..., description="The ID of the room to join"),
    auth_key: ApiKeyModel = Depends(get_api_key),  # <--- SECURED BY API KEY
):
    """
    Exchanges a valid API Key for a short-lived WebSocket Ticket (JWT).
    """
    logging_utility.info(
        f"SDK Client '{auth_key.user_id}' requesting WS ticket for {room_id}"
    )

    # 1. (Optional) Validation: Can this API Key access this room?
    # if not access_control.can_access(auth_key.user_id, room_id):
    #     raise HTTPException(403, "Access Denied")

    # 2. Mint the Ticket (valid for 30-60 seconds)
    # The ticket encodes the User ID and Room ID
    ticket = mint_computer_ticket(user_id=auth_key.user_id, room_id=room_id)

    return {"room_id": room_id, "token": ticket, "ws_url": SANDBOX_URL}
