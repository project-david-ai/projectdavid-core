# src/api/sandbox/services/shell_session.py
import asyncio
import logging
import mimetypes
import os
import pty
import shutil
import subprocess  # nosec B404
import termios
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from projectdavid import Entity
from sandbox.services.room_manager import RoomManager

load_dotenv()

logger = logging.getLogger("shell_session")

# ── Configuration ──────────────────────────────────────────────────────────────

# How long (seconds) a session may sit completely idle before auto-cleanup.
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes

# Base directory for per-thread session working dirs.
# Each session gets its own subdirectory: /app/sessions/{room}
# This is also the firejail --private root so file writes land here by default.
SESSIONS_BASE_DIR = "/app/sessions"

# Files larger than this are skipped during harvest to avoid flooding the
# file server with accidental bulk data.
MAX_HARVEST_FILE_SIZE_MB = 50

# Netfilter rules file applied by firejail.
# Blocks all RFC-1918 outbound (Docker-internal) while allowing the public
# internet.  See docker/sandbox/computer_shell_netfilter.conf.
NETFILTER_CONF = "/app/computer_shell_netfilter.conf"

# Set DISABLE_FIREJAIL=true in .env to skip sandboxing in local development.
_DISABLE_FIREJAIL = os.getenv("DISABLE_FIREJAIL", "false").lower() == "true"

# Set COMPUTER_SHELL_ALLOW_NET=true to bypass even the netfilter rules.
# Only needed for debugging — in normal operation the netfilter handles
# the internet-yes / internal-Docker-no distinction automatically.
_ALLOW_NET_UNRESTRICTED = (
    os.getenv("COMPUTER_SHELL_ALLOW_NET", "false").lower() == "true"
)

# ProjectDavid file server — mirrors the code_interpreter upload pattern.
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
_API_BASE_URL = os.getenv("SANDBOX_API_BASE_URL", "http://fastapi_cosmic_catalyst:9000")


# ── Firejail helpers ───────────────────────────────────────────────────────────


def _build_shell_cmd(session_dir: str, elevated: bool) -> list[str]:
    """
    Returns the argv list used to spawn the PTY shell.

    Security model
    ──────────────
    * --private=<session_dir>
        Sets the process HOME to session_dir and bind-mounts it, so relative
        file writes land there by default.  System paths (/bin, /usr, etc.) are
        still readable — full filesystem isolation via --overlay is a future
        hardening step.

    * --caps.drop=all + --seccomp + --nogroups
        Strips the most dangerous privilege-escalation vectors.

    * Network isolation  (the interesting bit)
        --net=eth0          Creates a new network namespace bridged to the
                            container's primary interface.  The shell process
                            gets its own namespace — other sessions and the
                            sandbox FastAPI process are unaffected.

        --netfilter=<path>  Applies iptables rules inside that namespace.
                            The rules (see computer_shell_netfilter.conf):
                              - ALLOW  loopback
                              - ALLOW  outbound DNS (udp/tcp 53)
                              - DROP   outbound to 10.0.0.0/8      (Docker bridge)
                              - DROP   outbound to 172.16.0.0/12   (Docker bridge)
                              - DROP   outbound to 192.168.0.0/16  (host-mode)
                              - ALLOW  everything else (public internet)

        Result: pip, curl, wget, public APIs all work.  Internal Docker
        services (mysql, redis, qdrant, api container) are unreachable.

        COMPUTER_SHELL_ALLOW_NET=true drops the --netfilter arg entirely,
        giving the shell unrestricted egress.  Only for debugging.

    * SYS_ADMIN capability
        --net=eth0 requires firejail to create a network namespace, which
        needs SYS_ADMIN.  The sandbox service already has cap_add: SYS_ADMIN
        in docker-compose, so no compose changes are needed.

    * Elevated sessions
        sudo /bin/bash is still supported; firejail wraps the sudo call so
        the elevated shell is still jailed to the session directory.
    """
    if _DISABLE_FIREJAIL:
        logger.warning(
            "DISABLE_FIREJAIL=true — computer shell running WITHOUT sandbox. "
            "Do not use in production."
        )
        return ["sudo", "/bin/bash"] if elevated else ["/bin/bash"]

    firejail_args = [
        "firejail",
        f"--private={session_dir}",
        "--noprofile",
        "--nogroups",
        "--nosound",
        "--notv",
        "--seccomp",
        "--caps.drop=all",
        # New network namespace bridged to the container's primary interface.
        # Required for --netfilter to apply rules in an isolated namespace
        # rather than affecting the whole container.
        "--net=eth0",
    ]

    if not _ALLOW_NET_UNRESTRICTED:
        # Apply RFC-1918 block rules — allows internet, blocks Docker-internal.
        firejail_args.append(f"--netfilter={NETFILTER_CONF}")
    else:
        logger.warning(
            "COMPUTER_SHELL_ALLOW_NET=true — netfilter bypassed for room. "
            "Internal Docker services are reachable from this shell."
        )

    inner_cmd = ["sudo", "/bin/bash"] if elevated else ["/bin/bash"]
    return firejail_args + inner_cmd


# ── Main session class ─────────────────────────────────────────────────────────


class PersistentShellSession:
    """
    One PTY-backed shell session per room (thread_id).

    What's new in this version
    ──────────────────────────
    1. Firejail sandboxing — each session is wrapped with capability drops,
       seccomp filtering, and per-process network namespace isolation.

    2. Network model — outbound internet allowed (pip, curl, wget, APIs).
       Docker-internal RFC-1918 ranges blocked via iptables netfilter rules
       applied inside a per-process network namespace.  COMPUTER_SHELL_ALLOW_NET
       and DISABLE_FIREJAIL env vars provide escape hatches for development.

    3. Per-thread working directory — /app/sessions/{room} is created at
       session start, passed to firejail as --private, and used as cwd.
       Files generated during the session accumulate there and are harvested
       on session end, mirroring the code_interpreter pattern exactly.

    4. File harvest & upload — on session end (idle timeout OR explicit
       disconnect) any files found in the session directory are uploaded to
       the ProjectDavid file server and broadcast to the room as computer_file
       events so the frontend can render download links.  The directory is
       then wiped.

    5. Explicit harvest_files action — the assistant can request a mid-session
       harvest via {"action": "harvest_files"} without terminating the shell.
       Useful after a long-running pipeline completes but the session should
       remain alive.

    Plus all improvements from the previous version:
    6. UUID sentinel per command — immune to output collision and split-read
       boundary issues.
    7. Non-blocking process reap via asyncio.to_thread.
    8. Idle timeout — auto-destructs after IDLE_TIMEOUT_SECONDS of inactivity.
    9. Session registry — registers with RoomManager so reconnects never spawn
       duplicate PTY processes.
    """

    def __init__(
        self,
        websocket: WebSocket,
        room: str,
        room_manager: RoomManager,
        elevated: bool = False,
    ):
        self.websocket = websocket
        self.room = room
        self.room_manager = room_manager
        self.elevated = elevated

        self.process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.alive = True

        # Per-thread working / output directory
        self.session_dir = os.path.join(SESSIONS_BASE_DIR, room)

        # Per-command sentinel state
        self._current_sentinel: Optional[str] = None
        self._read_buffer = ""

        # Idle timeout handle
        self._idle_timer: Optional[asyncio.TimerHandle] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.room_manager.connect(self.room, self.websocket)
        await self.room_manager.register_session(self.room, self)

        self._reset_idle_timer()

        # Create the per-session working directory.
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info("Session directory created: %s", self.session_dir)

        self.master_fd, slave_fd = pty.openpty()

        # Disable PTY echo so the sentinel wrapper never appears in output.
        try:
            attrs = termios.tcgetattr(slave_fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        except Exception as exc:
            logger.warning("Could not disable PTY echo: %s", exc)

        shell_cmd = _build_shell_cmd(self.session_dir, self.elevated)
        logger.info("Spawning shell for room %s: %s", self.room, shell_cmd)

        try:
            self.process = subprocess.Popen(
                shell_cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                preexec_fn=os.setsid,
                shell=False,
                close_fds=True,
                # shell=False is intentionally secure; PTY handles I/O directly
                cwd=self.session_dir,  # nosec B603
            )
            os.close(slave_fd)

            loop = asyncio.get_running_loop()
            loop.add_reader(self.master_fd, self._on_pty_readable)

            await self._message_loop()

        except WebSocketDisconnect:
            logger.info("Client disconnected from room %s", self.room)
        except Exception as exc:
            logger.error(
                "Unexpected error in shell session for room %s: %s", self.room, exc
            )
        finally:
            await self.cleanup()

    async def _message_loop(self) -> None:
        while self.alive:
            raw = await self.websocket.receive_text()
            import json

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = message.get("action")

            if action == "shell_command":
                self._reset_idle_timer()
                await self.send_command(message.get("command", ""))

            elif action == "ping":
                await self.websocket.send_json({"type": "pong"})

            elif action == "harvest_files":

                # Explicit mid-session harvest requested by the assistant.

                # Does NOT terminate the session — shell stays alive.

                self._reset_idle_timer()

                await self._harvest_and_upload_files(
                    context="explicit_harvest",
                    wipe_after=True,
                )

                # FIX: Broadcast command_complete so the SDK knows the harvest is done!

                asyncio.create_task(
                    self.room_manager.broadcast(
                        self.room,
                        {
                            "type": "command_complete",
                            "thread_id": self.room,
                        },
                    )
                )

            elif action == "disconnect":
                break

            else:
                await self.websocket.send_json(
                    {"type": "error", "content": f"Unknown action: {action}"}
                )

    # ── PTY reader ─────────────────────────────────────────────────────────────

    def _on_pty_readable(self) -> None:
        """
        Called by the event loop whenever the PTY master fd has data.
        Buffers reads and scans for the current sentinel token.
        """
        try:
            if not self.master_fd:
                return

            raw = os.read(self.master_fd, 4096)
            if not raw:
                self.alive = False
                return

            chunk = raw.decode("utf-8", errors="replace")
            self._read_buffer += chunk

            # ── Sentinel detection ────────────────────────────────────────────
            sentinel = self._current_sentinel
            completion_detected = False

            if sentinel and sentinel in self._read_buffer:
                completion_detected = True
                self._read_buffer = self._read_buffer.replace(sentinel, "")
                self._read_buffer = self._read_buffer.replace("\n\r\n", "\n")
                self._current_sentinel = None

            # Hold back enough chars that a split sentinel can be detected
            # on the next read.
            if sentinel:
                safe_len = max(0, len(self._read_buffer) - len(sentinel))
                to_send, self._read_buffer = (
                    self._read_buffer[:safe_len],
                    self._read_buffer[safe_len:],
                )
            else:
                to_send, self._read_buffer = self._read_buffer, ""

            if to_send:
                asyncio.create_task(
                    self.room_manager.broadcast(
                        self.room,
                        {
                            "type": "shell_output",
                            "thread_id": self.room,
                            "content": to_send,
                        },
                    )
                )

            if completion_detected:
                if self._read_buffer:
                    asyncio.create_task(
                        self.room_manager.broadcast(
                            self.room,
                            {
                                "type": "shell_output",
                                "thread_id": self.room,
                                "content": self._read_buffer,
                            },
                        )
                    )
                    self._read_buffer = ""

                logger.info("Command completion detected in room %s", self.room)
                asyncio.create_task(
                    self.room_manager.broadcast(
                        self.room,
                        {
                            "type": "command_complete",
                            "thread_id": self.room,
                        },
                    )
                )

        except OSError:
            self.alive = False

    # ── Command dispatch ───────────────────────────────────────────────────────

    async def send_command(self, cmd: str) -> None:
        """
        Injects a command into the PTY with a unique UUID sentinel so we know
        exactly when it finishes, then manually echoes the clean command to
        the room so the UI looks natural.
        """
        if not (self.master_fd and self.alive):
            return

        sentinel = f"###CMD_DONE_{uuid.uuid4().hex}###"
        self._current_sentinel = sentinel

        try:
            # Visual echo: show the clean command in the terminal
            asyncio.create_task(
                self.room_manager.broadcast(
                    self.room,
                    {
                        "type": "shell_output",
                        "thread_id": self.room,
                        "content": f"{cmd}\r\n",
                    },
                )
            )

            # Actual execution: command + sentinel echo
            wrapped = f"{cmd}; echo '{sentinel}'\n"
            os.write(self.master_fd, wrapped.encode())

        except OSError as exc:
            logger.error("Failed to write to PTY in room %s: %s", self.room, exc)
            self.alive = False

    # ── File harvest ───────────────────────────────────────────────────────────

    async def _harvest_and_upload_files(
        self,
        context: str = "session_end",
        wipe_after: bool = True,
    ) -> None:
        """
        Scans the session directory for regular files, uploads each to the
        ProjectDavid file server, and broadcasts a computer_file event per file
        so the frontend can render a download link.

        Mirrors the code_interpreter _upload_generated_files pattern exactly,
        adapted for the persistent shell context.

        Parameters
        ──────────
        context
            Label included in log lines and the broadcast event so the
            frontend can distinguish mid-session harvests from final ones.
            Values: "explicit_harvest" | "idle_timeout" | "session_end"

        wipe_after
            Delete successfully uploaded files after upload.  Always True in
            normal operation; False only for debugging.
        """
        # 1. Scan BOTH the room's session directory AND the global generated_files directory
        directories_to_check = [self.session_dir, "/app/generated_files"]

        candidates: list[tuple[str, str]] = []
        seen_paths = set()

        for target_dir in directories_to_check:
            if not os.path.isdir(target_dir):
                continue

            for fname in os.listdir(target_dir):
                fpath = os.path.join(target_dir, fname)

                # Prevent double-processing just in case paths overlap
                abs_path = os.path.abspath(fpath)
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)

                if not os.path.isfile(fpath):
                    continue

                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                if size_mb > MAX_HARVEST_FILE_SIZE_MB:
                    logger.warning(
                        "Skipping %s (%.1f MB exceeds %d MB limit)",
                        fname,
                        size_mb,
                        MAX_HARVEST_FILE_SIZE_MB,
                    )
                    continue
                candidates.append((fname, fpath))

        if not candidates:
            logger.info("Harvest [%s] room=%s: no files found.", context, self.room)
            return

        logger.info(
            "Harvest [%s] room=%s: uploading %d file(s).",
            context,
            self.room,
            len(candidates),
        )

        if not _ADMIN_API_KEY:
            logger.error(
                "ADMIN_API_KEY not set — cannot upload harvested files for room %s.",
                self.room,
            )
            return

        try:
            client = Entity(api_key=_ADMIN_API_KEY, base_url=_API_BASE_URL)
        except Exception as exc:
            logger.error("Failed to init ProjectDavid client for harvest: %s", exc)
            return

        async def _upload_one(fname: str, fpath: str) -> Optional[dict]:
            try:
                upload = await asyncio.to_thread(
                    client.files.upload_file,
                    file_path=fpath,
                    purpose="assistants",
                )
                signed_url = await asyncio.to_thread(
                    client.files.get_signed_url,
                    file_id=upload.id,
                    use_real_filename=True,
                )
                return {
                    "filename": fname,
                    "id": upload.id,
                    "url": signed_url,
                    "mime_type": (
                        mimetypes.guess_type(fname)[0] or "application/octet-stream"
                    ),
                }
            except Exception as exc:
                logger.error("Upload failed for %s: %s", fname, exc)
                return None

        results = await asyncio.gather(
            *[_upload_one(fn, fp) for fn, fp in candidates],
            return_exceptions=True,
        )

        uploaded_paths: list[str] = []
        for (fname, fpath), res in zip(candidates, results):
            if not isinstance(res, dict) or not res:
                continue

            uploaded_paths.append(fpath)

            # Broadcast a download-link event.
            # Shape mirrors code_interpreter_file so the frontend renderer
            # handles it without new code on that side.
            await self.room_manager.broadcast(
                self.room,
                {
                    "type": "computer_file",
                    "context": context,
                    "thread_id": self.room,
                    "filename": res["filename"],
                    "file_id": res["id"],
                    "url": res["url"],
                    "mime_type": res["mime_type"],
                },
            )

        logger.info(
            "Harvest [%s] room=%s: %d/%d file(s) uploaded.",
            context,
            self.room,
            len(uploaded_paths),
            len(candidates),
        )

        if wipe_after:
            for fpath in uploaded_paths:
                try:
                    # We only delete the specific files we successfully uploaded!
                    os.remove(fpath)
                except Exception:
                    pass

    # ── Idle timeout ───────────────────────────────────────────────────────────

    def _reset_idle_timer(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
        loop = asyncio.get_event_loop()
        self._idle_timer = loop.call_later(IDLE_TIMEOUT_SECONDS, self._on_idle_timeout)

    def _on_idle_timeout(self) -> None:
        logger.info(
            "Room %s idle for %ss — harvesting files and cleaning up.",
            self.room,
            IDLE_TIMEOUT_SECONDS,
        )
        asyncio.create_task(self._idle_cleanup())

    async def _idle_cleanup(self) -> None:
        """Harvest first, then tear down — called only from the idle timeout path."""
        await self._harvest_and_upload_files(context="idle_timeout", wipe_after=True)
        await self.cleanup(skip_harvest=True)  # harvest already done above

    # ── Cleanup ────────────────────────────────────────────────────────────────

    async def cleanup(self, skip_harvest: bool = False) -> None:
        """
        Tears down the PTY process, removes the session directory, and
        disconnects the WebSocket.

        Parameters
        ──────────
        skip_harvest
            Pass True when called from _idle_cleanup to avoid a double-upload.
            All other callers leave this False so a final harvest runs
            automatically on explicit disconnect.
        """
        if not self.alive and not self.master_fd and not self.process:
            return  # already cleaned up

        self.alive = False

        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

        # Final harvest on explicit disconnect.
        # _idle_cleanup passes skip_harvest=True so this is never a double-upload.
        if not skip_harvest:
            await self._harvest_and_upload_files(context="session_end", wipe_after=True)

        # Remove the session directory entirely.
        if os.path.isdir(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
                logger.info("Session directory removed: %s", self.session_dir)
            except Exception as exc:
                logger.warning(
                    "Could not remove session directory %s: %s",
                    self.session_dir,
                    exc,
                )

        loop = asyncio.get_running_loop()

        if self.master_fd is not None:
            try:
                loop.remove_reader(self.master_fd)
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None

        if self.process is not None:
            proc = self.process
            self.process = None
            try:
                proc.terminate()
                # Non-blocking wait — never stalls the event loop
                await asyncio.to_thread(proc.wait, timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        try:
            await self.room_manager.unregister_session(self.room, self)
            await self.room_manager.disconnect(self.room, self.websocket)
            await self.websocket.close()
        except Exception:
            pass

        logger.info("Session fully cleaned up for room %s", self.room)
