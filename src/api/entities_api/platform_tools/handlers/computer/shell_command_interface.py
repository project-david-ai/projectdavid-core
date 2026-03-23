# entities_api/platform_tools/handlers/computer/shell_command_interface.py
import os
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from entities_api.platform_tools.handlers.computer.shell_command_client import (
    run_commands,
    run_commands_sync,
)

from src.api.entities_api.services.logging_service import LoggingUtility

load_dotenv()
logging_utility = LoggingUtility()


class ShellCommandInterface:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        thread_id: Optional[str] = None,
        idle_timeout: float = 2.0,
    ):
        self.logging_utility = LoggingUtility()
        self.endpoint = endpoint or os.getenv("SHELL_SERVER_URL", "ws://localhost:8000/ws/computer")
        self.default_thread_id = thread_id or "thread_default_id"
        self.idle_timeout = idle_timeout

    async def run_commands_async_stream(
        self,
        commands: List[str],
        token: str,
        thread_id: Optional[str] = None,
        elevated: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Native Async Generator.
        Calls the underlying WebSocket client and yields chunks as they arrive.
        """
        target_room = thread_id or self.default_thread_id
        self.logging_utility.info(f"Executing async stream on room: {target_room}")

        # CRITICAL CHANGE: "async for" instead of "await"
        # We iterate over the chunks coming from the client
        async for chunk in run_commands(commands, target_room, token=token, elevated=elevated):
            yield chunk

    def run_commands(
        self,
        commands: List[str],
        token: str,
        thread_id: Optional[str] = None,
        elevated: bool = False,
        idle_timeout: Optional[float] = None,
    ) -> str:
        """Keep for backward compatibility with pure sync parts of the app."""
        target_room = thread_id or self.default_thread_id
        return run_commands_sync(commands, target_room, token=token, elevated=elevated)


# --- NEW ASYNC ENTRY POINT ---
async def run_shell_commands_async(
    commands: List[str],
    token: str,
    thread_id: Optional[str] = None,
    elevated: bool = False,
    endpoint: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    service = ShellCommandInterface(endpoint=endpoint, thread_id=thread_id)
    async for chunk in service.run_commands_async_stream(
        commands, token=token, thread_id=thread_id, elevated=elevated
    ):
        yield chunk
