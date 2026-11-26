"""Sandbox admission and delegated artifact ownership regression tests."""

import time
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi import HTTPException

sandbox_auth = import_module("src.api.entities_api.routers.sandbox_auth_router")
files = import_module("src.api.entities_api.routers.files_router")

SECRET = "sandbox-test-secret-for-hs256-only"


def ticket(**claims):
    now = int(time.time())
    return jwt.encode(
        {"iat": now, "exp": now + 60, **claims}, SECRET, algorithm="HS256"
    )


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setenv("SANDBOX_AUTH_SECRET", SECRET)
    db = MagicMock()
    run = SimpleNamespace(
        user_id="user_owner", thread_id="thread_1", status="in_progress"
    )
    admin = SimpleNamespace(is_admin=True)
    thread = SimpleNamespace(owner_id="user_owner", participants=[])
    records = {
        (sandbox_auth.Run, "run_1"): run,
        (sandbox_auth.User, "admin"): admin,
        (sandbox_auth.Thread, "thread_1"): thread,
    }
    db.get.side_effect = lambda model, identifier: records.get((model, identifier))
    fresh = MagicMock()
    fresh.get.return_value = SimpleNamespace(id="user_owner")
    factory = MagicMock()
    factory.return_value.__enter__.return_value = fresh
    monkeypatch.setattr(sandbox_auth, "SessionLocal", factory)
    receipt = MagicMock(return_value=None)
    monkeypatch.setattr(sandbox_auth, "erasure_state", receipt)
    tracked = MagicMock()
    monkeypatch.setattr(sandbox_auth, "track_authenticated_request", tracked)
    return SimpleNamespace(
        db=db,
        run=run,
        admin=admin,
        thread=thread,
        fresh=fresh,
        receipt=receipt,
        tracked=tracked,
    )


def admit(h, token, **kwargs):
    request = sandbox_auth.SandboxAdmissionRequest(token=token, **kwargs)
    return sandbox_auth.admit_sandbox_session(
        request, h.db, SimpleNamespace(user_id="admin")
    )


def test_run_identity_resolves_to_actual_owner(admission):
    h = admission
    result = admit(h, ticket(sub="run_run_1", scopes=["execution"]), mode="execute")
    assert result == {"owner_id": "user_owner"}
    h.tracked.assert_called_once_with("user_owner")
    h.receipt.assert_called_once_with(h.fresh, "user_owner")


def test_direct_user_ticket_checks_quarantine(admission):
    h = admission
    h.receipt.return_value = "pending"
    with pytest.raises(HTTPException) as exc:
        admit(h, ticket(sub="user_owner", type="sandbox_access"), mode="execute")
    assert exc.value.status_code == 403
    h.tracked.assert_called_once_with("user_owner")


@pytest.mark.parametrize(
    "case",
    [
        "missing_run",
        "terminal_run",
        "wrong_room",
        "observer_as_executor",
        "non_admin",
        "bad_signature",
    ],
)
def test_denied_admission(admission, case):
    h = admission
    token = ticket(sub="run_run_1", scopes=["execution", "shell"], room="thread_1")
    params = {"mode": "computer", "room": "thread_1", "role": "executor"}
    if case == "missing_run":
        token = ticket(
            sub="run_run_missing", scopes=["execution", "shell"], room="thread_1"
        )
    elif case == "terminal_run":
        h.run.status = "cancelled"
    elif case == "wrong_room":
        params["room"] = "thread_2"
    elif case == "observer_as_executor":
        token = ticket(
            sub="user_owner", type="ws_ticket", room="thread_1", scopes=["observation"]
        )
    elif case == "non_admin":
        h.admin.is_admin = False
    elif case == "bad_signature":
        token = jwt.encode(
            {
                "sub": "user_owner",
                "type": "sandbox_access",
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
            },
            "wrong-test-key-not-the-real-signing-key",
            algorithm="HS256",
        )
    with pytest.raises(HTTPException) as exc:
        admit(h, token, **params)
    assert exc.value.status_code == 403
    h.tracked.assert_not_called()


def test_observer_ticket_is_room_scoped(admission):
    h = admission
    token = ticket(
        sub="user_owner", type="ws_ticket", room="thread_1", scopes=["observation"]
    )
    assert admit(h, token, mode="computer", room="thread_1", role="viewer") == {
        "owner_id": "user_owner"
    }
    h.thread.owner_id = "other"
    with pytest.raises(HTTPException):
        admit(h, token, mode="computer", room="thread_1", role="viewer")


@pytest.fixture
def upload(monkeypatch):
    db = MagicMock()
    db.get.return_value = SimpleNamespace(is_admin=True)
    fresh = MagicMock()
    fresh.get.return_value = SimpleNamespace(id="user_owner")
    factory = MagicMock()
    factory.return_value.__enter__.return_value = fresh
    monkeypatch.setattr(files, "SessionLocal", factory)
    receipt = MagicMock(return_value=None)
    monkeypatch.setattr(files, "erasure_state", receipt)
    track = MagicMock()
    monkeypatch.setattr(files, "track_authenticated_request", track)
    service = MagicMock()
    monkeypatch.setattr(files, "FileService", service)
    response = MagicMock()
    response.model_validate.return_value = SimpleNamespace(
        id="file_1", filename="result.txt"
    )
    monkeypatch.setattr(files, "FileResponse", response)
    monkeypatch.setenv("SIGNED_URL_SECRET", "test-secret")
    monkeypatch.setenv("DOWNLOAD_BASE_URL", "https://example.invalid")
    return SimpleNamespace(
        db=db, fresh=fresh, receipt=receipt, track=track, service=service
    )


def test_artifact_is_persisted_as_resolved_owner(upload):
    h = upload
    file = MagicMock()
    result = files.upload_sandbox_artifact(
        "user_owner", file, h.db, SimpleNamespace(user_id="admin")
    )
    kwargs = h.service.return_value.upload_file.call_args.kwargs
    assert kwargs["file"] is file
    assert kwargs["request"].user_id == "user_owner"
    assert kwargs["request"].purpose == "assistants"
    assert result["id"] == "file_1"
    assert result["url"].startswith("https://example.invalid/v1/files/download?")
    h.track.assert_called_once_with("user_owner")
    h.receipt.assert_called_once_with(h.fresh, "user_owner")


@pytest.mark.parametrize("case", ["pending", "complete", "missing_owner", "non_admin"])
def test_artifact_upload_fails_closed(upload, case):
    h = upload
    if case == "non_admin":
        h.db.get.return_value.is_admin = False
    elif case == "missing_owner":
        h.fresh.get.return_value = None
    else:
        h.receipt.return_value = case
    with pytest.raises(HTTPException) as exc:
        files.upload_sandbox_artifact(
            "user_owner", MagicMock(), h.db, SimpleNamespace(user_id="admin")
        )
    assert exc.value.status_code == 403
    h.service.assert_not_called()


@pytest.fixture
def sandbox_client():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src/api/sandbox/services/core_artifacts.py"
    )
    spec = spec_from_file_location("sandbox_core_artifacts_under_test", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspaces_are_owner_scoped_and_unique(sandbox_client, tmp_path):
    from pathlib import Path

    one = Path(sandbox_client.create_workspace(str(tmp_path), "user_a"))
    two = Path(sandbox_client.create_workspace(str(tmp_path), "user_a"))
    three = Path(sandbox_client.create_workspace(str(tmp_path), "user_b"))
    assert one.parent == two.parent
    assert one != two
    assert one.parent != three.parent
    assert one.parent.parent == tmp_path


def test_sandbox_upload_transmits_resolved_owner_and_mime(
    sandbox_client, tmp_path, monkeypatch
):
    import asyncio

    import httpx

    artifact = tmp_path / "result.txt"
    artifact.write_text("hello")
    seen = []

    async def handle(request):
        seen.append(request)
        body = await request.aread()
        assert request.url.path == "/v1/internal/sandbox/artifacts"
        assert b"user_owner" in body
        assert b"hello" in body
        assert b"Content-Type: text/plain" in body
        return httpx.Response(201, json={"id": "file_1", "url": "signed"})

    monkeypatch.setattr(
        sandbox_client,
        "_connection",
        lambda: httpx.AsyncClient(
            base_url="http://core", transport=httpx.MockTransport(handle)
        ),
    )
    assert asyncio.run(
        sandbox_client.upload_artifact("user_owner", artifact.name, str(artifact))
    ) == {"id": "file_1", "url": "signed"}
    assert len(seen) == 1


def test_sandbox_admission_never_falls_back_when_core_denies(
    sandbox_client, monkeypatch
):
    import asyncio

    import httpx

    def handle(request):
        return httpx.Response(403, json={"detail": "Sandbox access denied."})

    monkeypatch.setattr(
        sandbox_client,
        "_connection",
        lambda: httpx.AsyncClient(
            base_url="http://core", transport=httpx.MockTransport(handle)
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(sandbox_client.admit_session("signed-ticket", mode="execute"))
