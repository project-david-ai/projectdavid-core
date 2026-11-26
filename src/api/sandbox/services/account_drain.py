"""Fail-closed drain for the dedicated, single-worker sandbox container.

All admission and background task registration happens on its event loop.
The synchronous drain has no await points; no socket can start in the gap
between checking for activity and removing an account's workspace.
"""

import asyncio
import hashlib
import os
import shutil
import stat
from functools import wraps
from pathlib import Path


class DrainBlocked(RuntimeError):
    pass


class SandboxActivity:
    def __init__(self):
        self.sockets = 0
        self.tasks = set()

    def track_socket(self, handler):
        @wraps(handler)
        async def wrapped(*args, **kwargs):
            # Count before the first await, including pending Core admission.
            self.sockets += 1
            try:
                return await handler(*args, **kwargs)
            finally:
                self.sockets -= 1

        return wrapped

    def create_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def is_busy(self):
        return bool(self.sockets or self.tasks)


activity = SandboxActivity()


def namespace_idle(proc_root=Path("/proc")):
    """Only accept the verified deployment shape: supervisor + one worker.

    Any extra PID, unreadable procfs, or different supervisor layout blocks
    erasure. This is not proof of isolation against a compromised host.
    """
    try:
        pid = os.getpid()
        if pid == 1 or os.getppid() != 1:
            return False
        if (proc_root / "1" / "comm").read_text().strip() != "supervisord":
            return False
        pids = {
            int(entry.name) for entry in proc_root.iterdir() if entry.name.isdecimal()
        }
        return pids == {1, pid}
    except (OSError, ValueError):
        return False


def _directory(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(mode):
        raise DrainBlocked("Unexpected workspace path type.")
    return True


def drain_account(owner_id, *, rooms_busy=False, roots=None):
    if not isinstance(owner_id, str) or not owner_id:
        raise DrainBlocked("Missing account owner.")
    if activity.is_busy() or rooms_busy or not namespace_idle():
        raise DrainBlocked("Sandbox is not quiescent.")
    if roots is None:
        roots = (Path("/app/account_executions"), Path("/app/sessions"))
    if not shutil.rmtree.avoids_symlink_attacks:
        raise DrainBlocked("Safe workspace removal is unavailable.")
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    paths = []
    # Validate all paths before mutating any of them. Roots are fixed by
    # the deployment, never supplied by a client.
    for base in roots:
        base = Path(base)
        if _directory(base):
            target = base / digest
            if _directory(target):
                paths.append(target)
    for target in paths:
        shutil.rmtree(target)
        if _directory(target):
            raise DrainBlocked("Workspace removal is incomplete.")
    if not namespace_idle():
        raise DrainBlocked("Sandbox quiescence changed during cleanup.")
