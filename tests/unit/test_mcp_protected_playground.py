"""Protected Playground authorization at the MCP management boundary."""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
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
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

mcp_router = import_module("src.api.entities_api.routers.mcp_router")


validator = ValidationInterface()
OWNER = "user_mcp_protection"
OTHER = "user_mcp_other"
PLAYGROUND = playground_assistant_id(OWNER)
NORMAL = "asst_mcp_normal"
OTHER_PROTECTED = "asst_mcp_other_protected"


@pytest.fixture
def state(monkeypatch, tmp_path):
    engine = real_create_engine(
        f"sqlite+pysqlite:///{(tmp_path / 'mcp-protection.db').as_posix()}",
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
            (OTHER_PROTECTED, True),
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
    service = McpRegistrationService(
        session_factory=factory,
        cache_invalidator_factory=lambda: SimpleNamespace(
            invalidate_sync=invalidated.append
        ),
    )
    server = service.register_server(
        validator.McpServerRegistrationCreate(
            name="test", url="https://example.com/mcp"
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
                selection={"server_id": server.id, "tools": ["echo"]},
            )
    finally:
        engine.dispose()


def _path(assistant_id):
    return f"/assistants/{assistant_id}/mcp-tools"


def _server_path(state):
    return f"/mcp/servers/{state.server.id}"


def _attach(state, assistant_id):
    response = state.client.post(_path(assistant_id), json=state.selection)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_ordinary_key_cannot_access_playground(state, method):
    kwargs = {} if method == "GET" else {"json": state.selection}
    response = state.client.request(method, _path(PLAYGROUND), **kwargs)
    assert response.status_code == 404
    assert response.json()["detail"] == "Assistant not found"
    state.discovery.assert_not_awaited()
    assert state.invalidated == []


def test_protected_key_can_manage_own_playground(state):
    state.key.is_protected = True
    attached = _attach(state, PLAYGROUND)
    assert len(attached) == 1
    assert state.client.get(_path(PLAYGROUND)).json() == attached
    assert (
        state.client.request(
            "DELETE", _path(PLAYGROUND), json=state.selection
        ).status_code
        == 204
    )
    assert state.client.get(_path(PLAYGROUND)).json() == []
    with state.factory() as db:
        assert db.get(Assistant, PLAYGROUND).tool_configs == []


@pytest.mark.parametrize("assistant_id", [PLAYGROUND, OTHER_PROTECTED])
def test_protected_key_is_not_a_global_bypass(state, assistant_id):
    state.key.is_protected = True
    if assistant_id == PLAYGROUND:
        state.key.user_id = OTHER
    assert state.client.get(_path(assistant_id)).status_code == 404
    assert (
        state.client.post(_path(assistant_id), json=state.selection).status_code == 404
    )
    state.discovery.assert_not_awaited()


def test_normal_assistant_and_unattached_registration_remain_manageable(state):
    _attach(state, NORMAL)
    assert len(state.client.get(_path(NORMAL)).json()) == 1
    assert (
        state.client.patch(
            _server_path(state), json={"timeout_seconds": 20}
        ).status_code
        == 200
    )
    assert (
        state.client.request("DELETE", _path(NORMAL), json=state.selection).status_code
        == 204
    )
    assert state.client.delete(_server_path(state)).status_code == 204


@pytest.mark.parametrize(
    "method,body",
    [
        ("PATCH", {"enabled": False}),
        ("PATCH", {"timeout_seconds": 20}),
        ("PATCH", {"name": "changed"}),
        ("DELETE", None),
    ],
)
def test_ordinary_key_cannot_mutate_shared_registration(state, method, body):
    _attach(state, NORMAL)
    state.key.is_protected = True
    _attach(state, PLAYGROUND)
    state.invalidated.clear()
    state.key.is_protected = False
    kwargs = {} if body is None else {"json": body}
    assert (
        state.client.request(method, _server_path(state), **kwargs).status_code == 403
    )
    # Registration metadata is user-scoped; only mutation is restricted.
    assert state.client.get(_server_path(state)).status_code == 200
    assert len(state.client.get(_path(NORMAL)).json()) == 1
    assert state.invalidated == []
    with state.factory() as db:
        row = db.get(McpServerRegistration, state.server.id)
        assert row.enabled is True
        assert row.name == state.server.name
        assert row.timeout_seconds == state.server.timeout_seconds
        assert db.query(AssistantMcpTool).count() == 2
        assert len(db.get(Assistant, PLAYGROUND).tool_configs) == 1
        assert len(db.get(Assistant, NORMAL).tool_configs) == 1


def test_protected_key_can_mutate_shared_registration(state):
    state.key.is_protected = True
    _attach(state, PLAYGROUND)
    _attach(state, NORMAL)
    assert (
        state.client.patch(
            _server_path(state), json={"timeout_seconds": 20}
        ).status_code
        == 200
    )
    assert state.client.delete(_server_path(state)).status_code == 204
    with state.factory() as db:
        assert db.query(AssistantMcpTool).count() == 0
        assert db.get(McpServerRegistration, state.server.id) is None
        assert db.get(Assistant, PLAYGROUND).tool_configs == []
        assert db.get(Assistant, NORMAL).tool_configs == []


def test_attachment_rechecks_authorization_after_discovery(state):
    async def discover(**kwargs):
        with state.factory() as db:
            db.get(Assistant, NORMAL).is_protected = True
            db.commit()
        return state.discovery.return_value

    state.discovery.side_effect = discover
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            state.service.attach_tools(
                assistant_id=NORMAL,
                attachment=validator.AssistantMcpToolsAttach(**state.selection),
                user_id=OWNER,
            )
        )
    assert error.value.status_code == 404
    with state.factory() as db:
        assert db.query(AssistantMcpTool).count() == 0
        assert db.get(Assistant, NORMAL).tool_configs == []
