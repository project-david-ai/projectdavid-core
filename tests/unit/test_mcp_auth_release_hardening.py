from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from cryptography.fernet import Fernet
from fastapi import HTTPException
from mcp.server import MCPServer
from projectdavid_common import ValidationInterface
from projectdavid_orm.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm.models import (
    Assistant,
    AssistantMcpTool,
    Credential,
    McpServerRegistration,
    User,
)
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.cli.docker_manager import DockerManager
from src.api.entities_api.orchestration.tool_abi import ToolCallEnvelope
from src.api.entities_api.routers.mcp_router import router as mcp_router
from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
    CredentialService,
)
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

validator = ValidationInterface()


class _NoopInvalidator:
    def invalidate_sync(
        self,
        assistant_id: str,
    ) -> None:
        return None


class _BearerGuard:
    def __init__(
        self,
        app: Any,
        *,
        token: str,
    ) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode("ascii")

    async def __call__(
        self,
        scope: Any,
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])

            if headers.get(b"authorization") != self.expected:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (
                                b"content-type",
                                b"text/plain",
                            )
                        ],
                    }
                )

                await send(
                    {
                        "type": "http.response.body",
                        "body": b"unauthorized",
                    }
                )

                return

        await self.app(
            scope,
            receive,
            send,
        )


@contextlib.asynccontextmanager
async def _serve(
    app: Any,
) -> AsyncIterator[str]:
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    sock.bind(("127.0.0.1", 0))

    sock.listen()
    sock.setblocking(False)

    port = sock.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="critical",
            lifespan="on",
        )
    )

    task = asyncio.create_task(server.serve(sockets=[sock]))

    try:
        for _ in range(500):
            if server.started:
                break

            if task.done():
                await task

            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Bearer MCP test server did not start")

        yield (f"http://127.0.0.1:{port}/mcp")
    finally:
        server.should_exit = True

        await asyncio.wait_for(
            task,
            timeout=5,
        )

        sock.close()


def _secured_mcp_server(
    token: str,
) -> _BearerGuard:
    server = MCPServer(
        name="auth1c-e2e",
        version="1.0.0",
    )

    @server.tool()
    async def echo(
        message: str,
    ) -> str:
        """Return an authenticated echo."""
        return f"secure:{message}"

    return _BearerGuard(
        server.streamable_http_app(),
        token=token,
    )


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker]:
    engine = create_engine(
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
        db.flush()

        db.add(User(id="user_2"))
        db.flush()

        db.add(
            Assistant(
                id="asst_auth1c",
                object="assistant",
                created_at=1,
                name="Auth-1C Assistant",
                owner_id="user_1",
                tool_configs=[],
            )
        )

        db.commit()

    try:
        yield factory
    finally:
        engine.dispose()


def _service(
    factory: sessionmaker,
    key: str | None,
) -> McpRegistrationService:
    credentials = CredentialService(
        session_factory=factory,
        key_provider=lambda: key,
    )

    return McpRegistrationService(
        session_factory=factory,
        cache_invalidator_factory=(lambda: _NoopInvalidator()),
        credential_service=credentials,
    )


def test_deployment_generates_valid_fernet_credential_root() -> None:
    key = DockerManager._generate_secret_value("PROJECT_DAVID_CREDENTIAL_KEY")

    Fernet(key.encode("ascii"))


def test_runtime_key_is_conditional_on_authenticated_registration(
    session_factory: sessionmaker,
) -> None:
    missing_key = CredentialService(
        session_factory=session_factory,
        key_provider=lambda: None,
    )

    # Empty installation: valid.
    missing_key.validate_runtime_configuration()

    public_service = _service(
        session_factory,
        Fernet.generate_key().decode("ascii"),
    )

    public_service.register_server(
        validator.McpServerRegistrationCreate(
            name="public",
            url=("https://public." "example.test/mcp"),
        ),
        user_id="user_1",
    )

    # Public-only installation: still valid.
    missing_key.validate_runtime_configuration()

    real_key = Fernet.generate_key().decode("ascii")

    protected_service = _service(
        session_factory,
        real_key,
    )

    protected_service.register_server(
        validator.McpServerRegistrationCreate(
            name="protected",
            url=("https://protected." "example.test/mcp"),
            auth={
                "type": "bearer",
                "token": "startup-token",
            },
        ),
        user_id="user_1",
    )

    with pytest.raises(
        CredentialConfigurationError,
    ):
        missing_key.validate_runtime_configuration()

    CredentialService(
        session_factory=session_factory,
        key_provider=lambda: real_key,
    ).validate_runtime_configuration()


def test_discovery_router_uses_typed_pydantic_response() -> None:
    route = next(
        route
        for route in mcp_router.routes
        if (
            getattr(
                route,
                "path",
                None,
            )
            == "/mcp/servers/{server_id}/tools"
            and "GET"
            in getattr(
                route,
                "methods",
                set(),
            )
        )
    )

    assert route.response_model is validator.McpToolDiscoveryPageRead

    page = validator.McpToolDiscoveryPageRead(
        tools=[
            validator.McpDiscoveredToolRead(
                server_id="mcpreg_1",
                remote_name="echo",
                canonical_id=("mcp:mcpreg_1:echo"),
                provider_name=("secure__echo"),
                definition={
                    "type": "function",
                    "function": {
                        "name": ("secure__echo"),
                        "parameters": {"type": "object"},
                    },
                },
            )
        ],
        next_cursor=None,
    )

    assert page.tools[0].remote_name == "echo"


def test_real_bearer_discovery_attachment_and_execution(
    session_factory: sessionmaker,
) -> None:
    async def exercise() -> None:
        token = "auth1c-real-loopback-secret"

        key = Fernet.generate_key().decode("ascii")

        service = _service(
            session_factory,
            key,
        )

        async with _serve(_secured_mcp_server(token)) as url:
            registration = service.register_server(
                validator.McpServerRegistrationCreate(
                    name=("secure-loopback"),
                    url=url,
                    auth={
                        "type": "bearer",
                        "token": token,
                    },
                ),
                user_id="user_1",
            )

            page = await service.discover_tools(
                user_id="user_1",
                server_id=registration.id,
            )

            assert [tool.remote_name for tool in page.tools] == ["echo"]

            attached = await service.attach_tools(
                assistant_id=("asst_auth1c"),
                attachment=(
                    validator.AssistantMcpToolsAttach(
                        server_id=(registration.id),
                        tools=["echo"],
                    )
                ),
                user_id="user_1",
            )

            assert len(attached) == 1

            executors = service.build_runtime_executors(assistant_id=("asst_auth1c"))

            assert len(executors) == 1

            result = await executors[0].execute(
                ToolCallEnvelope(
                    name=(executors[0].provider_name),
                    arguments={
                        "message": "hello",
                    },
                    run_id="run_auth1c",
                    thread_id=("thread_auth1c"),
                    assistant_id=("asst_auth1c"),
                    tool_call_id=("call_auth1c"),
                )
            )

            assert result.is_error is False
            assert result.content == "secure:hello"

            # Same endpoint, second tenant,
            # intentionally wrong credential.
            bad_service = _service(
                session_factory,
                key,
            )

            bad_registration = bad_service.register_server(
                validator.McpServerRegistrationCreate(
                    name="wrong-token",
                    url=url,
                    auth={
                        "type": "bearer",
                        "token": ("definitely-wrong"),
                    },
                ),
                user_id="user_2",
            )

            with pytest.raises(
                HTTPException,
            ) as exc_info:
                await bad_service.discover_tools(
                    user_id="user_2",
                    server_id=(bad_registration.id),
                )

            assert exc_info.value.status_code == 502

            assert "definitely-wrong" not in str(exc_info.value.detail)

            with session_factory() as db:
                row = db.get(
                    McpServerRegistration,
                    registration.id,
                )

                assert row is not None
                assert row.credential_id is not None

                credential = db.get(
                    Credential,
                    row.credential_id,
                )

                assert credential is not None
                assert credential.kind == "bearer"

                assert token not in credential.encrypted_payload

    asyncio.run(exercise())


def test_auth_downgrade_drops_credential_fk_before_index() -> None:
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "d3f8a1c6b904_bind_mcp_registration_auth.py"
    )

    source = migration_path.read_text(encoding="utf-8")

    downgrade = source[source.index("def downgrade() -> None:") :]

    assert downgrade.index("drop_fk_if_exists(") < downgrade.index(
        "drop_index_if_exists("
    )


def test_auth1a_downgrade_drops_owner_fk_before_indexes() -> None:
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "a4d7e6b9c2f1_add_tenant_credentials.py"
    )

    source = migration_path.read_text(encoding="utf-8")

    assert '_OWNER_FK_NAME = "fk_credentials_owner_id"' in source

    assert "name=_OWNER_FK_NAME" in source

    downgrade = source[source.index("def downgrade() -> None:") :]

    assert downgrade.index("drop_fk_if_exists(") < downgrade.index(
        '"ix_credentials_owner_id"'
    )
