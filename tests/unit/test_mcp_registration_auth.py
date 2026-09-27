from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from mcp.types import ListToolsResult
from projectdavid_common import ValidationInterface
from projectdavid_orm.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm.models import (
    Assistant,
    AssistantMcpTool,
    Credential,
    McpServerRegistration,
    User,
)
from sqlalchemy.engine.create import create_engine as real_create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Self

from src.api.entities_api.services.credential_service import CredentialService
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

validator = ValidationInterface()


class _FakeMcpClient:
    def __init__(self, result: ListToolsResult) -> None:
        self.result = result

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> ListToolsResult:
        assert cursor is None
        return self.result


class _NoopInvalidator:
    def invalidate_sync(self, assistant_id: str) -> None:
        return None


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker]:
    engine = real_create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
    )

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

    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    with factory() as db:
        db.add(User(id="user_1"))

        db.add(
            Assistant(
                id="asst_1",
                object="assistant",
                created_at=1,
                name="Auth Test Assistant",
                owner_id="user_1",
                tool_configs=[],
            )
        )

        db.commit()

    try:
        yield factory
    finally:
        engine.dispose()


def _remote_result() -> ListToolsResult:
    return ListToolsResult.model_validate(
        {
            "tools": [
                {
                    "name": "search_issues",
                    "description": "Search issues",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ]
        }
    )


def _service(
    session_factory: sessionmaker,
    key: str | None,
    captured_http_clients: list,
) -> McpRegistrationService:
    result = _remote_result()

    def client_factory(*args, **kwargs):
        captured_http_clients.append(kwargs.get("http_client"))
        return _FakeMcpClient(result)

    credential_service = CredentialService(
        session_factory=session_factory,
        key_provider=lambda: key,
    )

    return McpRegistrationService(
        session_factory=session_factory,
        client_factory=client_factory,
        cache_invalidator_factory=lambda: _NoopInvalidator(),
        credential_service=credential_service,
    )


def _bearer_request(token: str):
    return validator.McpServerRegistrationCreate(
        name="Authenticated MCP",
        url="https://mcp.example.test/mcp",
        auth={
            "type": "bearer",
            "token": token,
        },
    )


def test_bearer_registration_persists_only_ciphertext_and_safe_metadata(
    session_factory: sessionmaker,
) -> None:
    token = "auth1b-super-secret-token"
    key = Fernet.generate_key().decode("ascii")
    captured: list = []

    service = _service(
        session_factory,
        key,
        captured,
    )

    request = _bearer_request(token)

    assert token not in repr(request)
    assert token not in request.model_dump_json()

    registration = service.register_server(
        request,
        user_id="user_1",
    )

    assert registration.auth_type == "bearer"
    assert not hasattr(registration, "credential_id")
    assert token not in registration.model_dump_json()

    with session_factory() as db:
        row = db.get(
            McpServerRegistration,
            registration.id,
        )

        assert row is not None
        assert row.auth_type == "bearer"
        assert row.credential_id is not None

        credential = db.get(
            Credential,
            row.credential_id,
        )

        assert credential is not None
        assert credential.owner_id == "user_1"
        assert credential.kind == "bearer"
        assert token not in credential.encrypted_payload

    resolved = service._credential_service.resolve(
        credential_id=row.credential_id,
        user_id="user_1",
        expected_kind="bearer",
    )

    assert resolved == {"token": token}


def test_bearer_discovery_uses_authorization_and_closes_http_client(
    session_factory: sessionmaker,
) -> None:
    token = "discovery-token"
    key = Fernet.generate_key().decode("ascii")
    captured: list = []

    service = _service(
        session_factory,
        key,
        captured,
    )

    registration = service.register_server(
        _bearer_request(token),
        user_id="user_1",
    )

    page = asyncio.run(
        service.discover_tools(
            user_id="user_1",
            server_id=registration.id,
        )
    )

    assert [tool.remote_name for tool in page.tools] == ["search_issues"]

    assert len(captured) == 1

    http_client = captured[0]

    assert http_client is not None
    assert http_client.headers.get("Authorization") == f"Bearer {token}"
    assert http_client.is_closed is True


def test_attachment_and_runtime_reuse_registration_credential(
    session_factory: sessionmaker,
) -> None:
    token = "runtime-token"
    key = Fernet.generate_key().decode("ascii")
    captured: list = []

    service = _service(
        session_factory,
        key,
        captured,
    )

    registration = service.register_server(
        _bearer_request(token),
        user_id="user_1",
    )

    asyncio.run(
        service.attach_tools(
            assistant_id="asst_1",
            attachment=validator.AssistantMcpToolsAttach(
                server_id=registration.id,
                tools=["search_issues"],
            ),
            user_id="user_1",
        )
    )

    assert len(captured) == 1
    assert captured[0].is_closed is True

    executors = service.build_runtime_executors(
        assistant_id="asst_1",
    )

    assert len(executors) == 1

    async def open_runtime_client() -> None:
        async with executors[0]._client_factory():
            pass

    asyncio.run(open_runtime_client())

    assert len(captured) == 2

    runtime_http_client = captured[1]

    assert runtime_http_client is not None
    assert runtime_http_client.headers.get("Authorization") == f"Bearer {token}"
    assert runtime_http_client.is_closed is True


def test_missing_master_key_fails_closed_and_rolls_back(
    session_factory: sessionmaker,
) -> None:
    service = _service(
        session_factory,
        None,
        [],
    )

    with pytest.raises(HTTPException) as exc_info:
        service.register_server(
            _bearer_request("never-persist-me"),
            user_id="user_1",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "MCP credential storage is unavailable"
    assert "PROJECT_DAVID_CREDENTIAL_KEY" not in str(exc_info.value.detail)

    with session_factory() as db:
        assert db.query(Credential).count() == 0
        assert db.query(McpServerRegistration).count() == 0


def test_existing_endpoint_cannot_silently_change_auth_type(
    session_factory: sessionmaker,
) -> None:
    key = Fernet.generate_key().decode("ascii")
    service = _service(
        session_factory,
        key,
        [],
    )

    service.register_server(
        validator.McpServerRegistrationCreate(
            name="Public MCP",
            url="https://mcp.example.test/mcp",
        ),
        user_id="user_1",
    )

    with pytest.raises(HTTPException) as exc_info:
        service.register_server(
            _bearer_request("different-security-state"),
            user_id="user_1",
        )

    assert exc_info.value.status_code == 409

    with session_factory() as db:
        assert db.query(Credential).count() == 0
        assert db.query(McpServerRegistration).count() == 1


def test_delete_removes_unreferenced_bound_credential(
    session_factory: sessionmaker,
) -> None:
    key = Fernet.generate_key().decode("ascii")
    service = _service(
        session_factory,
        key,
        [],
    )

    registration = service.register_server(
        _bearer_request("delete-me"),
        user_id="user_1",
    )

    service.delete_server(
        server_id=registration.id,
        user_id="user_1",
    )

    with session_factory() as db:
        assert db.query(McpServerRegistration).count() == 0
        assert db.query(Credential).count() == 0
