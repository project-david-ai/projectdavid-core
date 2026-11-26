import hmac
import json
import os

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel
from sandbox.services.account_drain import DrainBlocked, activity, drain_account
from sandbox.services.code_execution_service import StreamingCodeExecutionHandler
from sandbox.services.core_artifacts import admit_session, verify_pending_erasure
from sandbox.services.logging_service import LoggingUtility
from sandbox.services.room_manager import RoomManager
from sandbox.services.shell_session import PersistentShellSession

# Load environment variables
load_dotenv()

logging_utility = LoggingUtility()
v1_router = APIRouter()
room_manager = RoomManager()


async def validate_token(
    websocket: WebSocket, token: str, *, mode="execute", room=None, role="executor"
) -> str | None:
    """Core validates the signed ticket and returns its current account owner."""
    try:
        return await admit_session(token, mode=mode, room=room, role=role)
    except Exception:
        logging_utility.warning("Sandbox admission rejected or unavailable")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


@v1_router.websocket("/execute")
@activity.track_socket
async def websocket_execute(
    websocket: WebSocket,
    token: str = Query(..., description="Short-lived JWT from Main API"),
):
    """
    Handles code execution via WebSocket.
    Secured via JWT Query parameter.
    """
    await websocket.accept()

    # 1. Security Check
    user_id = await validate_token(websocket, token)
    if not user_id:
        return  # Connection closed inside validate_token

    logging_utility.info(f"Execution connection established for User: {user_id}")

    try:
        data = await websocket.receive_text()
        parsed_data = json.loads(data)

        # Validate input
        if "code" not in parsed_data:
            await websocket.send_json({"error": "Missing required field: code"})
            await websocket.close(code=1003)
            return

        # 2. Execute with Trusted Identity
        # We ignore 'user_id' in parsed_data and use the one from the JWT
        handler = StreamingCodeExecutionHandler(owner_id=user_id)
        await handler.execute_code_streaming(
            websocket=websocket,
            code=parsed_data["code"],
            user_id=user_id,
        )

    except json.JSONDecodeError:
        await websocket.send_json({"error": "Invalid JSON input"})
        await websocket.close(code=1003)
    except KeyError as e:
        await websocket.send_json({"error": f"Missing required field: {str(e)}"})
        await websocket.close(code=1003)
    except WebSocketDisconnect:
        logging_utility.info(f"Client {user_id} disconnected")
    except Exception as e:
        logging_utility.error(f"Unexpected error: {str(e)}")
        await websocket.send_json({"error": "Internal server error"})
        await websocket.close(code=1011)


@v1_router.websocket("/computer")
@activity.track_socket
async def websocket_endpoint(
    websocket: WebSocket,
    room: str = Query(..., description="The Thread/Room ID"),
    elevated: bool = Query(False),
    token: str = Query(..., description="Signed JWT from Main API"),
    role: str = Query(
        "executor", description="Role of the client (viewer or executor)"
    ),
):
    """
    SECURED: Interactive Shell Session.
    """
    await websocket.accept()

    user_id = await validate_token(
        websocket, token, mode="computer", room=room, role=role
    )
    if user_id is None:
        return

    # 3. Handle Frontend UI (Viewer)
    if role == "viewer":
        logging_utility.info(f"Viewer UI connected to Room: {room}")
        await room_manager.connect(room, websocket)
        try:
            # Keep the connection alive and listen for the client disconnecting
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await room_manager.disconnect(room, websocket)
            logging_utility.info(f"Viewer UI disconnected from Room: {room}")
        return

    # 4. Handle AI SDK (Executor)
    logging_utility.info(f"Shell session started for User: {user_id} in Room: {room}")
    session = PersistentShellSession(
        websocket, room, room_manager, elevated=elevated, owner_id=user_id
    )
    await session.start()


class AccountDrainRequest(BaseModel):
    owner_id: str


@v1_router.post("/internal/account-drain", include_in_schema=False)
async def account_drain_endpoint(
    request: AccountDrainRequest, x_api_key: str = Header(default="")
):
    key = os.getenv("ADMIN_API_KEY")
    if not key or not hmac.compare_digest(key.encode(), x_api_key.encode()):
        raise HTTPException(status_code=403, detail="Sandbox drain denied.")
    try:
        await verify_pending_erasure(request.owner_id)
    except Exception:
        raise HTTPException(
            status_code=503, detail="Cannot verify pending erasure."
        ) from None
    try:
        drain_account(
            request.owner_id,
            rooms_busy=bool(room_manager.sessions or room_manager.connections),
        )
    except (DrainBlocked, OSError):
        raise HTTPException(status_code=409, detail="Sandbox drain pending.") from None
    return {"owner_id": request.owner_id, "drained": True}
