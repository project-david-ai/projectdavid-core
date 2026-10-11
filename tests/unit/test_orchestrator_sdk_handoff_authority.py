from __future__ import annotations

from src.api.entities_api.orchestration.engine.orchestrator_core import (
    _should_handoff_to_sdk_user_tool,
)


def test_unauthorized_unbound_function_is_not_sdk_handoff() -> None:
    assert (
        _should_handoff_to_sdk_user_tool(
            tool_name="playwright-primary__browser_click",
            platform_tools={"file_search"},
            mcp_executor=None,
            authorized_consumer_tool_names=frozenset(
                {
                    "jobs_delegate",
                    "application_prepare",
                }
            ),
        )
        is False
    )


def test_authorized_consumer_function_is_sdk_handoff() -> None:
    assert (
        _should_handoff_to_sdk_user_tool(
            tool_name="jobs_delegate",
            platform_tools={"file_search"},
            mcp_executor=None,
            authorized_consumer_tool_names=frozenset(
                {
                    "jobs_delegate",
                    "application_prepare",
                }
            ),
        )
        is True
    )


def test_platform_function_is_never_sdk_handoff() -> None:
    assert (
        _should_handoff_to_sdk_user_tool(
            tool_name="file_search",
            platform_tools={"file_search"},
            mcp_executor=None,
            authorized_consumer_tool_names=frozenset(
                {
                    "file_search",
                }
            ),
        )
        is False
    )


def test_bound_mcp_function_is_never_sdk_handoff() -> None:
    executor = object()

    assert (
        _should_handoff_to_sdk_user_tool(
            tool_name="playwright-primary__browser_snapshot",
            platform_tools={"file_search"},
            mcp_executor=executor,
            authorized_consumer_tool_names=frozenset(
                {
                    "playwright-primary__browser_snapshot",
                }
            ),
        )
        is False
    )
