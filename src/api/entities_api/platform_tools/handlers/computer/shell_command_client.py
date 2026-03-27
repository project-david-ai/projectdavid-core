#
"""
shell_command_client.py
────────────────────────
WebSocket client that drives the sandbox shell from the orchestration layer.

Changes in this version
────────────────────────
* computer_file events are no longer silently dropped.  They are serialised
  back to JSON and yielded as typed chunks so shell_execution_mixin.py can
  detect them, resolve signed URLs, and surface download links — exactly
  mirroring how code_interpreter_mixin.py handles code_interpreter_file events.

* harvest_files action support: send_harvest() sends the explicit harvest
  trigger to the sandbox and drains until the companion command_complete (the
  sandbox broadcasts computer_file events then a command_complete marker).

All previous fixes retained:
* Sequential command execution — send one, wait for command_complete, then next.
* No asyncio.sleep race conditions.
"""
import asyncio
import json
import logging
import os
from typing import AsyncGenerator, List

import websockets
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShellClient")

SHELL_SERVER_URL = os.getenv("SHELL_SERVER_URL", "ws://localhost:8000/ws/computer")


class ShellClient:
    def __init__(
        self,
        endpoint: str,
        room: str,
        token: str,
        elevated: bool = False,
        timeout: int = 60,
    ):
        self.endpoint = endpoint
        self.room = room
        self.token = token
        self.elevated = elevated
        self.timeout = timeout
        self.ws = None

    async def __aenter__(self) -> "ShellClient":
        conn_str = (
            f"{self.endpoint}"
            f"?room={self.room}"
            f"&elevated={str(self.elevated).lower()}"
            f"&token={self.token}"
        )
        logger.info("Connecting: %s", conn_str)
        self.ws = await websockets.connect(conn_str, ping_interval=self.timeout)
        logger.info("Connected to room '%s'", self.room)
        return self

    async def __aexit__(self, *_) -> None:
        if self.ws:
            try:
                await self.ws.send(json.dumps({"action": "disconnect"}))
            except Exception:
                pass
            await self.ws.close()
            logger.info("WebSocket closed for room '%s'", self.room)

    # ── Drain helpers ─────────────────────────────────────────────────────────

    async def _drain_until_complete(self) -> AsyncGenerator[str, None]:
        """
        Yield chunks until a command_complete message is received.

        Three chunk types are forwarded to the caller:

        1. shell_output / shell_error  — raw text, yielded as-is.
        2. computer_file               — yielded as a JSON string with a
                                         "type": "computer_file" discriminator
                                         so the mixin can detect and process it.
        3. command_complete            — signals end of this command; return.

        Everything else (pong, unknown) is silently consumed here.
        """
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type in ("shell_output", "shell_error"):
                content = data.get("content", "")
                if content:
                    logger.debug("Output chunk: %s", content.rstrip())
                    yield content

            elif msg_type == "computer_file":
                # Pass through as a typed JSON string.
                # The plain AsyncGenerator[str, None] interface is preserved —
                # the mixin detects these by attempting json.loads and checking
                # for the "computer_file" type key, exactly as code_interpreter
                # mixin does for "code_interpreter_file".
                logger.info(
                    "computer_file received: %s", data.get("filename", "<unknown>")
                )
                yield json.dumps({"type": "computer_file", **data})

            elif msg_type == "command_complete":
                logger.info("command_complete received")
                return

            elif msg_type == "pong":
                pass  # heartbeat reply — ignore

            else:
                logger.debug("Unrecognised message type '%s': %s", msg_type, data)

    # ── Command streaming ─────────────────────────────────────────────────────

    async def _stream(self, commands: List[str]) -> AsyncGenerator[str, None]:
        """
        Executes commands sequentially, yielding output chunks as they arrive.

        Protocol per command:
          1. Send  {"action": "shell_command", "command": <cmd>}
          2. Stream shell_output / computer_file chunks to caller
          3. Wait for command_complete before sending the next command
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected.")

        for idx, cmd in enumerate(commands):
            logger.info("Sending command %d/%d: %s", idx + 1, len(commands), cmd)
            await self.ws.send(json.dumps({"action": "shell_command", "command": cmd}))

            try:
                async for chunk in self._drain_until_complete():
                    yield chunk
            except Exception as exc:
                logger.error(
                    "Error while receiving output for command '%s': %s", cmd, exc
                )
                yield f"\n[Connection Error: {exc}]\n"
                return

        logger.info("All %d command(s) executed.", len(commands))

    # ── Harvest trigger ───────────────────────────────────────────────────────

    async def harvest_files(self) -> AsyncGenerator[str, None]:
        """
        Sends {"action": "harvest_files"} to the sandbox and drains the
        resulting computer_file events until command_complete.

        The sandbox shell_session treats harvest_files identically to a
        shell_command in terms of the response protocol: it broadcasts
        computer_file events then a command_complete marker, so
        _drain_until_complete handles it without modification.
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected.")

        logger.info("Sending harvest_files to room '%s'", self.room)
        await self.ws.send(json.dumps({"action": "harvest_files"}))

        async for chunk in self._drain_until_complete():
            yield chunk

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(self, commands: List[str]) -> AsyncGenerator[str, None]:
        """Primary public interface — returns an async generator of output chunks."""
        return self._stream(commands)

    async def execute(self, commands: List[str]) -> str:
        """Convenience wrapper — collects all output into a single string."""
        buf = ""
        async for chunk in self._stream(commands):
            buf += chunk
        return buf


# ── Module-level helpers ──────────────────────────────────────────────────────


async def run_commands(
    commands: List[str],
    room: str,
    token: str,
    elevated: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Async generator: open a client, stream all commands, trigger a final
    harvest, then close cleanly.

    The harvest step runs after all commands complete so any files written
    during the session are captured before the WebSocket closes.  Files are
    yielded as computer_file JSON chunks alongside normal text output — the
    mixin downstream distinguishes them by type.
    """
    async with ShellClient(SHELL_SERVER_URL, room, token, elevated) as client:
        # Execute all commands, forwarding output + any mid-session file events
        async for chunk in client.stream(commands):
            yield chunk

        # Explicit end-of-call harvest — guarantees files written during this
        # tool call are captured before the connection closes.
        async for chunk in client.harvest_files():
            yield chunk


def run_commands_sync(
    commands: List[str],
    room: str,
    token: str,
    elevated: bool = False,
) -> str:
    """Blocking wrapper for sync call-sites."""

    async def _collect() -> str:
        buf = ""
        async for chunk in run_commands(commands, room, token, elevated):
            buf += chunk
        return buf

    return asyncio.run(_collect())
