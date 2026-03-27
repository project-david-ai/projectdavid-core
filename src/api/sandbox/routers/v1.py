import json
import os

import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sandbox.services.code_execution_service import StreamingCodeExecutionHandler
from sandbox.services.logging_service import LoggingUtility
from sandbox.services.room_manager import RoomManager
from sandbox.services.shell_session import PersistentShellSession

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SANDBOX_AUTH_SECRET")
ALGORITHM = "HS256"

logging_utility = LoggingUtility()
v1_router = APIRouter()
room_manager = RoomManager()


async def validate_token(websocket: WebSocket, token: str) -> str | None:
    """
    Validates the JWT token from the WebSocket query parameter.
    Returns the user_id if valid, None otherwise.
    Closes the WebSocket with a policy violation error if invalid.
    """
    if not token:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token"
        )
        return None

    try:
        # Decode and verify the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if not user_id:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token payload"
            )
            return None

        return str(user_id)

    except jwt.ExpiredSignatureError:
        logging_utility.warning("Connection rejected: Token expired")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Token expired"
        )
        return None
    except jwt.InvalidTokenError:
        logging_utility.warning("Connection rejected: Invalid token")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token"
        )
        return None
    except Exception as e:
        logging_utility.error(f"Token validation error: {str(e)}")
        await websocket.close(
            code=status.WS_1011_INTERNAL_ERROR, reason="Authentication error"
        )
        return None


@v1_router.websocket("/execute")
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
    handler = StreamingCodeExecutionHandler()

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

    # 1. Validate Token Signature
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception as e:
        logging_utility.warning(f"Connection rejected: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Validate Room Access (Multi-Tenancy Security)
    allowed_room = payload.get("room")
    user_id = payload.get("sub")

    if allowed_room != room and allowed_room != "*":
        logging_utility.warning(
            f"Unauthorized Room Access: User {user_id} tried to join {room} but ticket was for {allowed_room}"
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
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
            await room_manager.disconnect(room, websocket)
            logging_utility.info(f"Viewer UI disconnected from Room: {room}")
        return

    # 4. Handle AI SDK (Executor)
    logging_utility.info(f"Shell session started for User: {user_id} in Room: {room}")
    session = PersistentShellSession(websocket, room, room_manager, elevated=elevated)
    await session.start()
