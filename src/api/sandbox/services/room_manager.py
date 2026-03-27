# src/api/sandbox/services/room_manager.py
import asyncio
import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

from fastapi import WebSocket

if TYPE_CHECKING:
    from sandbox.services.shell_session import PersistentShellSession

logger = logging.getLogger("room_manager")


class RoomManager:
    """
    Manages WebSocket connections and shell session lifecycle per room.

    Key invariant: at most one PersistentShellSession is alive per room at
    any given time.  When a new WebSocket joins a room that already has a
    live session, the old session is torn down cleanly before the new one
    starts, preventing PTY / fd leaks and double-broadcast races.
    """

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.sessions: Dict[str, "PersistentShellSession"] = {}
        self._lock = asyncio.Lock()

    # ── WebSocket registration ────────────────────────────────────────────

    async def connect(self, room: str, websocket: WebSocket) -> None:
        async with self._lock:
            if room not in self.connections:
                self.connections[room] = set()
            self.connections[room].add(websocket)

    async def disconnect(self, room: str, websocket: WebSocket) -> None:
        async with self._lock:
            if room in self.connections:
                self.connections[room].discard(websocket)
                if not self.connections[room]:
                    del self.connections[room]

    # ── Session registry ──────────────────────────────────────────────────

    async def register_session(
        self, room: str, session: "PersistentShellSession"
    ) -> None:
        """
        Register a new session for a room.  If a stale session already exists
        for this room, tear it down first so we never have two PTY processes
        broadcasting to the same room simultaneously.
        """
        async with self._lock:
            existing = self.sessions.get(room)

        if existing is not None and existing is not session:
            logger.warning(
                "Room %s already has a live session — tearing down stale session "
                "before registering the new one.",
                room,
            )
            await existing.cleanup()

        async with self._lock:
            self.sessions[room] = session
        logger.info("Session registered for room %s", room)

    async def unregister_session(
        self, room: str, session: "PersistentShellSession"
    ) -> None:
        """Remove a session from the registry (only if it is still the current one)."""
        async with self._lock:
            if self.sessions.get(room) is session:
                del self.sessions[room]
                logger.info("Session unregistered for room %s", room)

    def get_session(self, room: str) -> Optional["PersistentShellSession"]:
        return self.sessions.get(room)

    # ── Broadcast ─────────────────────────────────────────────────────────

    async def broadcast(self, room: str, message: dict) -> None:
        async with self._lock:
            sockets = self.connections.get(room, set()).copy()

        if not sockets:
            return

        coros = [
            ws.send_json(message)
            for ws in sockets
            if ws.client_state.name == "CONNECTED"
        ]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
