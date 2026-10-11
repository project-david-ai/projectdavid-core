from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace

from src.api.entities_api.orchestration.mcp_run_session_manager import (
    McpRunSessionManager,
)
from src.api.entities_api.orchestration.tool_abi import ToolResultEnvelope


class FakeClient:
    def __init__(self, identity: int) -> None:
        self.identity = identity


class FakeContext(AbstractAsyncContextManager):
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    async def __aenter__(self) -> FakeClient:
        self.state["enters"] += 1
        return FakeClient(self.state["enters"])

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.state["exits"] += 1


class FakeExecutor:
    def __init__(
        self,
        *,
        server_id: str,
        provider_name: str,
        state: dict[str, int],
    ) -> None:
        self.server_id = server_id
        self.provider_name = provider_name
        self.state = state

    def client_context(self) -> FakeContext:
        return FakeContext(self.state)

    async def execute_connected(
        self,
        client: FakeClient,
        call,
        *,
        progress_callback=None,
    ) -> ToolResultEnvelope:
        return ToolResultEnvelope(content=f"{call.name}:{client.identity}")

    def failure_result(self, exc: Exception) -> ToolResultEnvelope:
        return ToolResultEnvelope(
            content=f"ERROR:{type(exc).__name__}:{exc}",
            is_error=True,
        )


def _call(
    *,
    name: str,
    run_id: str,
):
    return SimpleNamespace(
        name=name,
        run_id=run_id,
    )


def test_same_tenant_run_registration_reuses_session_across_tools() -> None:
    async def scenario() -> None:
        state = {"enters": 0, "exits": 0}
        manager = McpRunSessionManager()

        navigate = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="navigate",
            state=state,
        )
        snapshot = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        first = await manager.execute(
            owner_scope="user_a",
            executor=navigate,
            call=_call(name="navigate", run_id="run_1"),
        )

        second = await manager.execute(
            owner_scope="user_a",
            executor=snapshot,
            call=_call(name="snapshot", run_id="run_1"),
        )

        assert first.content == "navigate:1"
        assert second.content == "snapshot:1"
        assert state == {"enters": 1, "exits": 0}

        await manager.close_run("run_1")

        assert state == {"enters": 1, "exits": 1}

    asyncio.run(scenario())


def test_different_runs_get_different_sessions() -> None:
    async def scenario() -> None:
        state = {"enters": 0, "exits": 0}
        manager = McpRunSessionManager()

        executor = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        first = await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_1"),
        )

        second = await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_2"),
        )

        assert first.content == "snapshot:1"
        assert second.content == "snapshot:2"
        assert state["enters"] == 2

        await manager.close_all()

        assert state["exits"] == 2

    asyncio.run(scenario())


def test_different_tenants_get_different_sessions() -> None:
    async def scenario() -> None:
        state = {"enters": 0, "exits": 0}
        manager = McpRunSessionManager()

        executor = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        first = await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_same"),
        )

        second = await manager.execute(
            owner_scope="user_b",
            executor=executor,
            call=_call(name="snapshot", run_id="run_same"),
        )

        assert first.content == "snapshot:1"
        assert second.content == "snapshot:2"
        assert state["enters"] == 2

        await manager.close_all()

    asyncio.run(scenario())


def test_different_registrations_get_different_sessions() -> None:
    async def scenario() -> None:
        state = {"enters": 0, "exits": 0}
        manager = McpRunSessionManager()

        first_executor = FakeExecutor(
            server_id="mcpreg_a",
            provider_name="snapshot_a",
            state=state,
        )
        second_executor = FakeExecutor(
            server_id="mcpreg_b",
            provider_name="snapshot_b",
            state=state,
        )

        first = await manager.execute(
            owner_scope="user_a",
            executor=first_executor,
            call=_call(name="snapshot_a", run_id="run_1"),
        )

        second = await manager.execute(
            owner_scope="user_a",
            executor=second_executor,
            call=_call(name="snapshot_b", run_id="run_1"),
        )

        assert first.content == "snapshot_a:1"
        assert second.content == "snapshot_b:2"
        assert state["enters"] == 2

        await manager.close_all()

    asyncio.run(scenario())


def test_close_one_run_does_not_close_another_run() -> None:
    async def scenario() -> None:
        state = {"enters": 0, "exits": 0}
        manager = McpRunSessionManager()

        executor = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_1"),
        )

        await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_2"),
        )

        assert state == {"enters": 2, "exits": 0}

        await manager.close_run("run_1")

        assert state == {"enters": 2, "exits": 1}

        again = await manager.execute(
            owner_scope="user_a",
            executor=executor,
            call=_call(name="snapshot", run_id="run_2"),
        )

        # run_2 kept its original session.
        assert again.content == "snapshot:2"
        assert state == {"enters": 2, "exits": 1}

        await manager.close_all()

        assert state == {"enters": 2, "exits": 2}

    asyncio.run(scenario())


def test_same_run_serializes_concurrent_calls_on_one_session() -> None:
    async def scenario() -> None:
        state = {
            "enters": 0,
            "exits": 0,
            "active": 0,
            "max_active": 0,
        }

        manager = McpRunSessionManager()

        class SerialProbeExecutor(FakeExecutor):
            async def execute_connected(
                self,
                client: FakeClient,
                call,
                *,
                progress_callback=None,
            ) -> ToolResultEnvelope:
                self.state["active"] += 1
                self.state["max_active"] = max(
                    self.state["max_active"],
                    self.state["active"],
                )

                try:
                    await asyncio.sleep(0.02)
                    return ToolResultEnvelope(content=f"{call.name}:{client.identity}")
                finally:
                    self.state["active"] -= 1

        navigate = SerialProbeExecutor(
            server_id="mcpreg_playwright",
            provider_name="navigate",
            state=state,
        )

        snapshot = SerialProbeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        first, second = await asyncio.gather(
            manager.execute(
                owner_scope="user_a",
                executor=navigate,
                call=_call(name="navigate", run_id="run_1"),
            ),
            manager.execute(
                owner_scope="user_a",
                executor=snapshot,
                call=_call(name="snapshot", run_id="run_1"),
            ),
        )

        assert first.content == "navigate:1"
        assert second.content == "snapshot:1"

        # One protocol session was created.
        assert state["enters"] == 1

        # The actor serialized tools/call requests on that connection.
        assert state["max_active"] == 1

        await manager.close_run("run_1")

        assert state["exits"] == 1

    asyncio.run(scenario())


def test_runtime_accessor_preserves_session_across_separate_handler_scopes() -> None:
    async def scenario() -> None:
        from src.api.entities_api.orchestration.mcp_run_session_manager import (
            get_mcp_run_session_manager,
        )

        state = {"enters": 0, "exits": 0}

        navigate = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="navigate",
            state=state,
        )

        snapshot = FakeExecutor(
            server_id="mcpreg_playwright",
            provider_name="snapshot",
            state=state,
        )

        # Simulate one provider/orchestrator request scope.
        first_handler_manager = get_mcp_run_session_manager()

        first = await first_handler_manager.execute(
            owner_scope="user_a",
            executor=navigate,
            call=_call(
                name="navigate",
                run_id="run_cross_request",
            ),
        )

        # Simulate the next HTTP inference continuation constructing a
        # completely new provider/orchestrator object.
        second_handler_manager = get_mcp_run_session_manager()

        assert second_handler_manager is first_handler_manager

        second = await second_handler_manager.execute(
            owner_scope="user_a",
            executor=snapshot,
            call=_call(
                name="snapshot",
                run_id="run_cross_request",
            ),
        )

        assert first.content == "navigate:1"
        assert second.content == "snapshot:1"

        # Only one underlying MCP protocol session existed.
        assert state["enters"] == 1
        assert state["exits"] == 0

        await second_handler_manager.close_run("run_cross_request")

        assert state["exits"] == 1

    asyncio.run(scenario())
