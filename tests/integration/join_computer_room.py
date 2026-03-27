import asyncio
import os

import websockets
from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv()

# --- Compatibility for different websockets versions ---
try:
    from websockets.exceptions import InvalidStatus
except ImportError:
    # Fallback for older versions
    from websockets.exceptions import InvalidStatusCode as InvalidStatus


async def verify_computer_connection():
    print("--- 1. Requesting Session Ticket via SDK ---")

    # --- Step 1: Get the Ticket via SDK ---
    try:

        client = Entity(api_key=os.getenv("ENTITIES_API_KEY"))
        session_data = client.computer.create_session(room_id="the_room_id")

        print(f"   Response: {session_data}")
    except Exception as e:
        print(f"❌ Failed to get ticket: {e}")
        return

    # --- Step 2: Build Connection URL ---
    # We use the URL provided by the SDK directly (no localhost substitution needed)
    base_ws_url = session_data["ws_url"]

    # Construct the full URL with Auth Params
    # Pattern: ws://[HOST]:[PORT]/ws/computer?room=...&token=...
    full_ws_url = (
        f"{base_ws_url}"
        f"?room={session_data['room_id']}"
        f"&token={session_data['token']}"
    )

    print(f"\n--- 2. Connecting to WebSocket ---\n   Target: {full_ws_url}")

    # --- Step 3: Connect and Maintain Session ---
    try:
        # ping_interval=None prevents the client from timing out if the server is quiet
        async with websockets.connect(full_ws_url, ping_interval=None) as websocket:
            print("\n✅ [SUCCESS] WebSocket Handshake Complete!")
            print("   (Authentication worked and Route was found)")

            # Optional: Send a ping to verify bidirectional flow
            # await websocket.send(json.dumps({"type": "ping"}))

            print("   Waiting for server activity (Ctrl+C to stop)...")

            while True:
                try:
                    # Wait up to 5 seconds for a message
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📩 Received: {message}")
                except asyncio.TimeoutError:
                    # No message received in 5 seconds.
                    # If we reach here, the connection is STABLE (heartbeat).
                    print("   ... connection is stable (heartbeat) ...")

                    # If you want to stop after a successful check, uncomment the break below:
                    # break
                except websockets.exceptions.ConnectionClosed as e:
                    print(
                        f"\n⚠️ Server closed connection: Code {e.code} (Reason: {e.reason})"
                    )
                    break

    except InvalidStatus as e:
        # Handle HTTP errors (403 Forbidden, 404 Not Found, etc.)
        status_code = (
            getattr(e, "response", None)
            and e.response.status_code
            or getattr(e, "status_code", "Unknown")
        )
        print(f"\n❌ Connection Rejected: Status {status_code}")

        if status_code == 404:
            print("   -> Route not found (Check URL path vs @websocket router)")
        elif status_code == 403:
            print("   -> Authentication failed (Check Token/API Key)")

    except websockets.exceptions.ConnectionClosedError:
        print(
            "\n❌ Connection lost unexpectedly (Server likely crashed or exited the endpoint function)."
        )

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(verify_computer_connection())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
