"""MCP-5B1 durable registration and assistant attachment tests."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from mcp.types import ListToolsResult
from projectdavid_common import ValidationInterface
from projectdavid_orm.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm.models import (
    Assistant,
    AssistantMcpTool,
    McpServerRegistration,
    User,
)
from sqlalchemy.engine.create import create_engine as real_create_engine
from sqlalchemy.orm import sessionmaker
from typing_extensions import Self

from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)
from src.api.entities_api.services.mcp_tool_config import (
    function_tool_name,
    merge_user_tools_preserving_managed,
)

validator = ValidationInterface()


class _CacheInvalidator:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    def invalidate_sync(self, assistant_id: str) -> None:
        self.invalidated.append(assistant_id)


class _FakeMcpClient:
    def __init__(
        self,
        result: ListToolsResult,
    ) -> None:
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
                name="Test Assistant",
                owner_id="user_1",
                tool_configs=[
                    {
                        "type": "function",
                        "function": {
                            "name": "consumer_search",
                            "description": "Consumer search",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
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
                        "properties": {
                            "query": {
                                "type": "string",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ]
        }
    )


def _service(
    session_factory: sessionmaker,
) -> tuple[McpRegistrationService, _CacheInvalidator]:
    invalidator = _CacheInvalidator()
    result = _remote_result()

    def client_factory(*args, **kwargs):
        return _FakeMcpClient(result)

    return (
        McpRegistrationService(
            session_factory=session_factory,
            client_factory=client_factory,
            cache_invalidator_factory=lambda: invalidator,
        ),
        invalidator,
    )


def test_registration_is_create_or_return_existing(
    session_factory: sessionmaker,
) -> None:
    service, _ = _service(session_factory)

    first = service.register_server(
        validator.McpServerRegistrationCreate(
            name="GitHub",
            url="https://MCP.EXAMPLE.test:443/mcp",
        ),
        user_id="user_1",
    )

    second = service.register_server(
        validator.McpServerRegistrationCreate(
            name="Different display name",
            url="https://mcp.example.test/mcp",
        ),
        user_id="user_1",
    )

    assert first.id == second.id
    assert first.name == "GitHub"
    assert second.name == "GitHub"

    with session_factory() as db:
        assert db.query(McpServerRegistration).count() == 1


def test_attach_is_idempotent_and_injects_function_tool(
    session_factory: sessionmaker,
) -> None:
    service, invalidator = _service(session_factory)

    registration = service.register_server(
        validator.McpServerRegistrationCreate(
            name="GitHub",
            url="https://mcp.example.test/mcp",
        ),
        user_id="user_1",
    )

    request = validator.AssistantMcpToolsAttach(
        server_id=registration.id,
        tools=["search_issues"],
    )

    first = asyncio.run(
        service.attach_tools(
            assistant_id="asst_1",
            attachment=request,
            user_id="user_1",
        )
    )

    second = asyncio.run(
        service.attach_tools(
            assistant_id="asst_1",
            attachment=request,
            user_id="user_1",
        )
    )

    assert first[0].id == second[0].id
    assert first[0].provider_name == second[0].provider_name
    assert first[0].canonical_id == (f"mcp:{registration.id}:search_issues")
    assert first[0].provider_name == "GitHub__search_issues"
    assert registration.id not in first[0].provider_name
    assert invalidator.invalidated == ["asst_1", "asst_1"]

    with session_factory() as db:
        assert db.query(AssistantMcpTool).count() == 1

        assistant = db.get(Assistant, "asst_1")
        assert assistant is not None

        names = [
            name
            for tool in assistant.tool_configs
            if (name := function_tool_name(tool)) is not None
        ]

        assert names.count("consumer_search") == 1
        assert names.count(first[0].provider_name) == 1

        mcp_tool = next(
            tool
            for tool in assistant.tool_configs
            if function_tool_name(tool) == first[0].provider_name
        )

        assert mcp_tool["function"]["description"] == "Search issues"
        assert mcp_tool["function"]["parameters"]["required"] == ["query"]


def test_detach_removes_provenance_and_model_capability(
    session_factory: sessionmaker,
) -> None:
    service, invalidator = _service(session_factory)

    registration = service.register_server(
        validator.McpServerRegistrationCreate(
            name="GitHub",
            url="https://mcp.example.test/mcp",
        ),
        user_id="user_1",
    )

    request = validator.AssistantMcpToolsAttach(
        server_id=registration.id,
        tools=["search_issues"],
    )

    attached = asyncio.run(
        service.attach_tools(
            assistant_id="asst_1",
            attachment=request,
            user_id="user_1",
        )
    )

    service.detach_tools(
        assistant_id="asst_1",
        attachment=validator.AssistantMcpToolsDetach(
            server_id=registration.id,
            tools=["search_issues"],
        ),
        user_id="user_1",
    )

    # Detach again: contractual no-op.
    service.detach_tools(
        assistant_id="asst_1",
        attachment=validator.AssistantMcpToolsDetach(
            server_id=registration.id,
            tools=["search_issues"],
        ),
        user_id="user_1",
    )

    with session_factory() as db:
        assert db.query(AssistantMcpTool).count() == 0

        assistant = db.get(Assistant, "asst_1")
        assert assistant is not None

        names = {
            name
            for tool in assistant.tool_configs
            if (name := function_tool_name(tool)) is not None
        }

        assert "consumer_search" in names
        assert attached[0].provider_name not in names

    assert invalidator.invalidated == ["asst_1", "asst_1"]


def test_generic_tool_replacement_preserves_managed_mcp_tools() -> None:
    managed = {
        "type": "function",
        "function": {
            "name": "mcpreg_123__search",
            "description": "Managed MCP search",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }

    incoming_consumer = {
        "type": "function",
        "function": {
            "name": "consumer_new",
            "description": "Consumer replacement",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }

    result = merge_user_tools_preserving_managed(
        [managed],
        [incoming_consumer],
        {"mcpreg_123__search"},
    )

    assert [function_tool_name(tool) for tool in result] == [
        "consumer_new",
        "mcpreg_123__search",
    ]


def test_generic_tool_update_may_round_trip_but_not_mutate_managed_spec() -> None:
    managed = {
        "type": "function",
        "function": {
            "name": "mcpreg_123__search",
            "description": "Managed MCP search",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }

    assert merge_user_tools_preserving_managed(
        [managed],
        [managed],
        {"mcpreg_123__search"},
    ) == [managed]

    modified = {
        **managed,
        "function": {
            **managed["function"],
            "description": "User-modified description",
        },
    }

    with pytest.raises(
        ValueError,
        match="cannot be modified",
    ):
        merge_user_tools_preserving_managed(
            [managed],
            [modified],
            {"mcpreg_123__search"},
        )


def test_registration_and_assistant_are_user_scoped(
    session_factory: sessionmaker,
) -> None:
    service, _ = _service(session_factory)

    registration = service.register_server(
        validator.McpServerRegistrationCreate(
            name="GitHub",
            url="https://mcp.example.test/mcp",
        ),
        user_id="user_1",
    )

    with pytest.raises(
        Exception,
    ) as exc_info:
        asyncio.run(
            service.attach_tools(
                assistant_id="asst_1",
                attachment=validator.AssistantMcpToolsAttach(
                    server_id=registration.id,
                    tools=["search_issues"],
                ),
                user_id="user_2",
            )
        )

    assert getattr(exc_info.value, "status_code", None) in {403, 404}
