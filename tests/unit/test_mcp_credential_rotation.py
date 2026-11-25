"""Explicit bearer rotation preserves MCP identity and authorization."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.types import Tool
from projectdavid_common import ValidationInterface
from projectdavid_orm.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm.models import (
    Assistant,
    AssistantMcpTool,
    Credential,
    McpServerRegistration,
    User,
)
from sqlalchemy.engine import Engine
from sqlalchemy.engine.create import create_engine as real_create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.orchestration.mcp_tool_discovery import adapt_mcp_tools
from src.api.entities_api.services.assistants_service import playground_assistant_id
from src.api.entities_api.services.credential_service import CredentialService
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

mcp_router = import_module("src.api.entities_api.routers.mcp_router")

validator = ValidationInterface()
OWNER = "user_mcp_rotation"
OTHER = "user_mcp_rotation_other"
PLAYGROUND = playground_assistant_id(OWNER)
NORMAL = "asst_mcp_normal"
OLD_TOKEN = "old-test-token"
NEW_TOKEN = "new-test-token"


@pytest.fixture
def state(monkeypatch, tmp_path):
    engine = real_create_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'mcp-rotation.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    assert isinstance(engine, Engine), "MCP fixture requires a real SQLite engine"
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Credential.__table__,
            Assistant.__table__,
            McpServerRegistration.__table__,
            AssistantMcpTool.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        for user_id in (OWNER, OTHER):
            db.add(User(id=user_id))
            db.flush()
        for identifier, protected in (
            (PLAYGROUND, True),
            (NORMAL, False),
        ):
            db.add(
                Assistant(
                    id=identifier,
                    owner_id=OWNER,
                    object="assistant",
                    created_at=1,
                    name="Test assistant",
                    tool_configs=[],
                    is_protected=protected,
                )
            )
            db.flush()
        db.commit()

    invalidated = []
    encryption_key = Fernet.generate_key().decode("ascii")
    credentials = CredentialService(
        session_factory=factory, key_provider=lambda: encryption_key
    )
    service = McpRegistrationService(
        credential_service=credentials,
        session_factory=factory,
        cache_invalidator_factory=lambda: SimpleNamespace(
            invalidate_sync=invalidated.append
        ),
    )
    server = service.register_server(
        validator.McpServerRegistrationCreate(
            name="test",
            url="https://example.com/mcp",
            auth={"type": "bearer", "token": OLD_TOKEN},
        ),
        user_id=OWNER,
    )
    discovery = AsyncMock(
        return_value=adapt_mcp_tools(
            server.id,
            [Tool(name="echo", inputSchema={"type": "object"})],
            provider_namespace="test",
        )
    )
    monkeypatch.setattr(service, "_discover_all_tools", discovery)
    monkeypatch.setattr(mcp_router, "McpRegistrationService", lambda: service)
    key = SimpleNamespace(user_id=OWNER, is_protected=False)
    app = FastAPI()
    app.include_router(mcp_router.router)
    app.dependency_overrides[mcp_router.get_api_key] = lambda: key
    app.dependency_overrides[mcp_router.get_db] = lambda: None
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(
                client=client,
                key=key,
                server=server,
                factory=factory,
                service=service,
                discovery=discovery,
                invalidated=invalidated,
                credentials=credentials,
                selection={"server_id": server.id, "tools": ["echo"]},
            )
    finally:
        engine.dispose()


def _path(state):
    return f"/mcp/servers/{state.server.id}/credential"


def _rotate(state, token=NEW_TOKEN):
    return state.client.put(_path(state), json={"type": "bearer", "token": token})


def _snapshot(state):
    with state.factory() as db:
        registration = db.get(McpServerRegistration, state.server.id)
        credential = db.get(Credential, registration.credential_id)
        return credential.id, credential.encrypted_payload


def _attach(state, assistant_id):
    response = state.client.post(
        f"/assistants/{assistant_id}/mcp-tools", json=state.selection
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_rotation_preserves_registration_credential_and_tools(state):
    attached = _attach(state, NORMAL)
    previous_id, previous_ciphertext = _snapshot(state)
    state.invalidated.clear()
    response = _rotate(state)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == state.server.id
    assert response.json()["auth_type"] == "bearer"
    assert response.json()["enabled"] is True
    assert NEW_TOKEN not in response.text
    assert previous_ciphertext not in response.text
    assert "credential_id" not in response.json()
    current_id, current_ciphertext = _snapshot(state)
    assert current_id == previous_id
    assert current_ciphertext != previous_ciphertext
    assert NEW_TOKEN not in current_ciphertext
    assert state.credentials.resolve(
        credential_id=current_id, user_id=OWNER, expected_kind="bearer"
    ) == {"token": NEW_TOKEN}
    assert state.client.get(f"/assistants/{NORMAL}/mcp-tools").json() == attached
    assert state.invalidated == [NORMAL]
    with state.factory() as db:
        assert db.query(Credential).count() == 1
        assert (
            db.get(Assistant, NORMAL).tool_configs[0]["function"]["name"]
            == attached[0]["provider_name"]
        )


def test_rotation_is_not_implicit_in_register(state):
    original = _snapshot(state)
    result = state.service.register_server(
        validator.McpServerRegistrationCreate(
            name="test",
            url="https://example.com/mcp",
            auth={"type": "bearer", "token": NEW_TOKEN},
        ),
        user_id=OWNER,
    )
    assert result.id == state.server.id
    assert _snapshot(state) == original


def test_protected_playground_rotation_requires_protected_key(state):
    state.key.is_protected = True
    attached = _attach(state, PLAYGROUND)
    original = _snapshot(state)
    state.key.is_protected = False
    assert _rotate(state).status_code == 403
    assert _snapshot(state) == original
    state.key.is_protected = True
    assert _rotate(state).status_code == 200
    assert state.client.get(f"/assistants/{PLAYGROUND}/mcp-tools").json() == attached


def test_cross_tenant_rotation_denied(state):
    original = _snapshot(state)
    state.key.user_id = OTHER
    state.key.is_protected = True
    assert _rotate(state).status_code == 404
    assert _snapshot(state) == original


@pytest.mark.parametrize(
    "body",
    [
        {"type": "bearer", "token": ""},
        {"type": "bearer", "token": "bad\r\nsecret"},
        {"type": "bearer", "token": {"secret": NEW_TOKEN}},
        {"type": "none", "token": NEW_TOKEN},
        {"type": "bearer", "token": NEW_TOKEN, "extra": NEW_TOKEN},
    ],
)
def test_invalid_rotation_does_not_echo_secret(state, body):
    original = _snapshot(state)
    response = state.client.put(_path(state), json=body)
    assert response.status_code == 422
    assert NEW_TOKEN not in response.text
    assert "bad" not in response.text
    assert _snapshot(state) == original


def test_rotation_rejects_unauthenticated_registration(state):
    server = state.service.register_server(
        validator.McpServerRegistrationCreate(
            name="public", url="https://example.com/public"
        ),
        user_id=OWNER,
    )
    response = state.client.put(
        f"/mcp/servers/{server.id}/credential",
        json={"type": "bearer", "token": NEW_TOKEN},
    )
    assert response.status_code == 409
    with state.factory() as db:
        assert db.get(McpServerRegistration, server.id).credential_id is None


def test_encryption_failure_does_not_replace_credential(state, monkeypatch):
    original = _snapshot(state)
    monkeypatch.setattr(state.credentials, "_key_provider", lambda: None)
    response = _rotate(state)
    assert response.status_code == 503
    assert NEW_TOKEN not in response.text
    assert _snapshot(state) == original


def test_rotation_does_not_replace_wrong_credential_kind(state):
    credential_id, original_ciphertext = _snapshot(state)
    with state.factory() as db:
        credential = db.get(Credential, credential_id)
        credential.kind = "inference_provider"
        db.commit()
    assert _rotate(state).status_code == 409
    assert _snapshot(state) == (credential_id, original_ciphertext)


def test_malformed_json_does_not_echo_secret(state):
    original = _snapshot(state)
    response = state.client.put(
        _path(state),
        content='{"type":"bearer","token":"' + NEW_TOKEN + '"',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert NEW_TOKEN not in response.text
    assert _snapshot(state) == original
