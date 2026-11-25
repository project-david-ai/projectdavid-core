"""Conservative sandbox drain: no inferred termination or cross-owner removal."""

import asyncio
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = spec_from_file_location(name, ROOT / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def drain(monkeypatch):
    mod = load("src/api/sandbox/services/account_drain.py", "sandbox_drain_test")
    monkeypatch.setattr(mod, "namespace_idle", lambda: True)
    # Windows host tests exercise policy; Linux runtime checks the real flag.
    monkeypatch.setattr(mod.shutil.rmtree, "avoids_symlink_attacks", True)
    return mod


def workspace(root, owner):
    path = root / hashlib.sha256(owner.encode()).hexdigest()
    path.mkdir(parents=True)
    (path / "artifact.txt").write_text("test")
    return path


def test_drain_only_target_and_is_idempotent(drain, tmp_path):
    roots = [tmp_path / "code", tmp_path / "shell"]
    targets = [workspace(root, "owner") for root in roots]
    others = [workspace(root, "other") for root in roots]
    drain.drain_account("owner", roots=roots)
    drain.drain_account("owner", roots=roots)
    assert all(not path.exists() for path in targets)
    assert all((path / "artifact.txt").is_file() for path in others)


@pytest.mark.parametrize("reason", ["socket", "task", "room", "process"])
def test_busy_drain_does_not_remove_data(drain, tmp_path, monkeypatch, reason):
    target = workspace(tmp_path, "owner")
    if reason == "socket":
        drain.activity.sockets = 1
    elif reason == "task":
        drain.activity.tasks.add(object())
    elif reason == "process":
        monkeypatch.setattr(drain, "namespace_idle", lambda: False)
    with pytest.raises(drain.DrainBlocked):
        drain.drain_account("owner", roots=[tmp_path], rooms_busy=reason == "room")
    assert target.exists()


def test_all_paths_validated_before_any_removal(drain, tmp_path):
    good = workspace(tmp_path / "good", "owner")
    bad = tmp_path / "bad"
    bad.write_text("not a directory")
    with pytest.raises(drain.DrainBlocked):
        drain.drain_account("owner", roots=[good.parent, bad])
    assert good.exists()


def test_socket_and_task_counted_until_completion(drain):
    async def scenario():
        release = asyncio.Event()
        entered = asyncio.Event()

        @drain.activity.track_socket
        async def connection():
            entered.set()
            await release.wait()

        socket = asyncio.create_task(connection())
        await entered.wait()
        task = drain.activity.create_task(release.wait())
        assert drain.activity.sockets == 1
        assert drain.activity.is_busy()
        release.set()
        await asyncio.gather(socket, task)
        await asyncio.sleep(0)
        assert not drain.activity.is_busy()

    asyncio.run(scenario())


def test_namespace_requires_only_supervisor_and_worker(tmp_path, monkeypatch):
    mod = load("src/api/sandbox/services/account_drain.py", "namespace_drain_test")
    monkeypatch.setattr(
        mod, "os", SimpleNamespace(getpid=lambda: 95, getppid=lambda: 1)
    )
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "comm").write_text("supervisord\n")
    (tmp_path / "95").mkdir()
    assert mod.namespace_idle(tmp_path)
    (tmp_path / "100").mkdir()
    assert not mod.namespace_idle(tmp_path)


@pytest.fixture
def client(monkeypatch):
    mod = load("src/api/entities_api/services/sandbox_drain.py", "drain_client_test")
    monkeypatch.setenv("ADMIN_API_KEY", "test-only-key")
    monkeypatch.setenv("SANDBOX_SERVER_URL", "http://sandbox:8000")
    return mod


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (200, {"owner_id": "owner", "drained": True}, None),
        (409, {}, 409),
        (503, {}, 503),
        (200, {"owner_id": "other", "drained": True}, 503),
        (200, {"owner_id": "owner", "drained": 1}, 503),
        (200, {"owner_id": "owner", "drained": False}, 503),
    ],
)
def test_core_requires_specific_positive_ack(
    client, monkeypatch, status, body, expected
):
    factory = httpx.Client

    def handle(request):
        assert request.url.path == "/ws/internal/account-drain"
        assert request.headers["X-API-Key"] == "test-only-key"
        return httpx.Response(status, json=body)

    monkeypatch.setattr(
        client.httpx,
        "Client",
        lambda **kwargs: factory(transport=httpx.MockTransport(handle), **kwargs),
    )
    if expected is None:
        client.require_sandbox_drain("owner")
    else:
        with pytest.raises(HTTPException) as exc:
            client.require_sandbox_drain("owner")
        assert exc.value.status_code == expected


def test_missing_configuration_blocks_core(client, monkeypatch):
    monkeypatch.delenv("SANDBOX_SERVER_URL")
    with pytest.raises(HTTPException) as exc:
        client.require_sandbox_drain("owner")
    assert exc.value.status_code == 503


@pytest.fixture
def route_harness(drain, monkeypatch):
    import sys
    from unittest.mock import AsyncMock, Mock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    verify = AsyncMock()
    rooms = SimpleNamespace(sessions={}, connections={})
    mocks = {
        "sandbox.services.account_drain": drain,
        "sandbox.services.core_artifacts": SimpleNamespace(
            admit_session=AsyncMock(), verify_pending_erasure=verify
        ),
        "sandbox.services.code_execution_service": SimpleNamespace(
            StreamingCodeExecutionHandler=Mock()
        ),
        "sandbox.services.logging_service": SimpleNamespace(LoggingUtility=Mock()),
        "sandbox.services.room_manager": SimpleNamespace(RoomManager=lambda: rooms),
        "sandbox.services.shell_session": SimpleNamespace(
            PersistentShellSession=Mock()
        ),
    }
    for name, module in mocks.items():
        monkeypatch.setitem(sys.modules, name, module)
    routes = load("src/api/sandbox/routers/v1.py", "sandbox_drain_routes_test")
    remove = Mock()
    monkeypatch.setattr(routes, "drain_account", remove)
    monkeypatch.setenv("ADMIN_API_KEY", "test-only-key")
    app = FastAPI()
    app.include_router(routes.v1_router, prefix="/ws")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, verify=verify, remove=remove, rooms=rooms)


@pytest.mark.parametrize(
    "case,expected",
    [("bad_key", 403), ("not_pending", 503), ("busy", 409), ("success", 200)],
)
def test_drain_route_authorization_and_ack(route_harness, drain, case, expected):
    h = route_harness
    if case == "not_pending":
        h.verify.side_effect = RuntimeError("not pending")
    elif case == "busy":
        h.remove.side_effect = drain.DrainBlocked("busy")
    response = h.client.post(
        "/ws/internal/account-drain",
        json={"owner_id": "owner"},
        headers={"X-API-Key": "bad" if case == "bad_key" else "test-only-key"},
    )
    assert response.status_code == expected
    if case == "bad_key":
        h.verify.assert_not_called()
    else:
        h.verify.assert_awaited_once_with("owner")
    if case in ("bad_key", "not_pending"):
        h.remove.assert_not_called()
    else:
        h.remove.assert_called_once_with("owner", rooms_busy=False)
    if case == "success":
        assert response.json() == {"owner_id": "owner", "drained": True}
