import os
import time
from typing import Literal

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import Run, Thread, User
from src.api.entities_api.services.erasure_state import PENDING, erasure_state
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.request_activity import track_authenticated_request
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


class SandboxAdmissionRequest(BaseModel):
    token: str
    mode: Literal["execute", "computer"]
    room: str | None = None
    role: Literal["viewer", "executor"] = "executor"


def _deny_sandbox():
    raise HTTPException(status_code=403, detail="Sandbox access denied.")


def _resolve_sandbox_owner(db: Session, request: SandboxAdmissionRequest) -> str:
    """Resolve only signed identities; never trust an owner from socket data."""
    secret = os.getenv("SANDBOX_AUTH_SECRET")
    if not secret:
        raise HTTPException(
            status_code=503, detail="Sandbox authentication unavailable."
        )
    try:
        claims = jwt.decode(
            request.token,
            secret,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.InvalidTokenError:
        _deny_sandbox()
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        _deny_sandbox()
    scopes = claims.get("scopes", [])
    if not isinstance(scopes, list):
        _deny_sandbox()
    kind = claims.get("type")
    if request.mode == "computer":
        if not request.room or claims.get("room") != request.room:
            _deny_sandbox()
    elif request.room is not None or request.role != "executor":
        _deny_sandbox()

    if kind == "sandbox_access":
        if request.mode != "execute":
            _deny_sandbox()
        owner = subject
    elif kind == "ws_ticket":
        if (
            request.mode != "computer"
            or request.role != "viewer"
            or "observation" not in scopes
        ):
            _deny_sandbox()
        owner = subject
        thread = db.get(Thread, request.room)
        if thread is None or (
            thread.owner_id != owner
            and not any(p.id == owner for p in thread.participants)
        ):
            _deny_sandbox()
    elif kind is None and subject.startswith("run_") and "execution" in scopes:
        if request.role != "executor" or (
            request.mode == "computer" and "shell" not in scopes
        ):
            _deny_sandbox()
        if request.mode == "execute" and (
            "shell" in scopes or claims.get("room") is not None
        ):
            _deny_sandbox()
        # Existing Core tokens prepend run_ to the entire persisted Run ID.
        run = db.get(Run, subject.removeprefix("run_"))
        if run is None or not run.user_id:
            _deny_sandbox()
        if str(getattr(run.status, "value", run.status)) not in {
            "in_progress",
            "action_required",
        }:
            _deny_sandbox()
        if request.mode == "computer" and run.thread_id != request.room:
            _deny_sandbox()
        owner = run.user_id
    else:
        _deny_sandbox()
    return owner


@router.post("/internal/sandbox/admission", include_in_schema=False)
def admit_sandbox_session(
    request: SandboxAdmissionRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    admin = db.get(User, auth_key.user_id)
    if admin is None or not admin.is_admin:
        _deny_sandbox()
    owner = _resolve_sandbox_owner(db, request)
    track_authenticated_request(owner)
    with SessionLocal() as admission_db:
        if (
            erasure_state(admission_db, owner) is not None
            or admission_db.get(User, owner) is None
        ):
            _deny_sandbox()
    return {"owner_id": owner}


class SandboxErasureRequest(BaseModel):
    owner_id: str


@router.post("/internal/sandbox/erasure-pending", include_in_schema=False)
def verify_sandbox_erasure_pending(
    request: SandboxErasureRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    admin = db.get(User, auth_key.user_id)
    if admin is None or not admin.is_admin:
        _deny_sandbox()
    with SessionLocal() as fresh:
        if erasure_state(fresh, request.owner_id) != PENDING:
            _deny_sandbox()
    return {"owner_id": request.owner_id, "pending": True}
