import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.entities_api.orchestration.mixins import web_search_mixin as web

URL = "https://docs.python.org/3/?query=value#section"
LABEL = "https://docs.python.org/3/"
ICON = "data:image/png;base64,AA=="  # Transport fixture, not image validation.


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(
        web,
        "ToolValidator",
        lambda: SimpleNamespace(validate_args=lambda *args: None),
    )
    instance = web.WebSearchMixin()
    instance._native_exec = SimpleNamespace(
        create_action=AsyncMock(return_value=SimpleNamespace(id="action_1")),
        read_url=AsyncMock(return_value="Python documentation"),
        scroll_url=AsyncMock(return_value="Page content"),
        search_url=AsyncMock(return_value="Keyword context"),
        get_web_favicon=AsyncMock(return_value=ICON),
        submit_tool_output=AsyncMock(),
        submit_failed_tool_execution=AsyncMock(),
        update_action_status=AsyncMock(),
    )
    return instance


def collect(worker, tool, args, call_id="call_1"):
    async def scenario():
        return [
            json.loads(event)
            async for event in worker._execute_web_tool_logic(
                tool_name=tool,
                required_keys=list(args),
                thread_id="thread_1",
                run_id="run_1",
                assistant_id="asst_1",
                arguments_dict=args,
                tool_call_id=call_id,
                decision=None,
            )
        ]

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    "tool,args",
    [
        ("read_web_page", {"url": URL}),
        ("scroll_web_page", {"url": URL, "page": 0}),
        ("search_web_page", {"url": URL, "query": "Python"}),
    ],
)
def test_page_status_metadata(worker, tool, args):
    events = collect(worker, tool, args)
    assert all(event["tool_call_id"] == "call_1" for event in events)
    assert all(event["origin_run_id"] == "run_1" for event in events)
    assert all(event["url"] == LABEL for event in events)
    assert all("favicon" not in event for event in events[:-1])
    assert events[-1]["status"] == "success"
    assert events[-1]["favicon"] == ICON
    worker._native_exec.get_web_favicon.assert_awaited_once_with(URL)
    output = worker._native_exec.submit_tool_output.call_args.kwargs
    assert ICON not in output["content"]
    assert output["is_error"] is False
    assert output["tool_call_id"] == "call_1"
    assert worker._native_exec.update_action_status.call_args.kwargs["status"] == (
        web.StatusEnum.completed.value
    )


def test_cache_failure_is_nonfatal(worker):
    worker._native_exec.get_web_favicon.side_effect = RuntimeError("cache unavailable")
    events = collect(worker, "search_web_page", {"url": URL, "query": "Python"})
    assert events[-1]["status"] == "success"
    assert "favicon" not in events[-1]
    assert worker._native_exec.submit_tool_output.call_args.kwargs["content"] == (
        "Keyword context"
    )
    assert worker._native_exec.submit_tool_output.call_args.kwargs["is_error"] is False


def test_distinct_invocations_have_distinct_identity(worker):
    args = {"url": URL, "query": "Python"}
    first = collect(worker, "search_web_page", args, "call_1")
    second = collect(worker, "search_web_page", args, "call_2")
    assert {event["tool_call_id"] for event in first} == {"call_1"}
    assert {event["tool_call_id"] for event in second} == {"call_2"}


def test_search_discovery_does_not_lookup_icons(worker, monkeypatch):
    monkeypatch.setattr(
        web,
        "SearxNGClient",
        lambda: SimpleNamespace(format_for_agent=AsyncMock(return_value="Results")),
    )
    events = collect(worker, "perform_web_search", {"query": "Python"})
    assert events[-1]["status"] == "success"
    assert all(event["tool_call_id"] == "call_1" for event in events)
    assert all("url" not in event and "favicon" not in event for event in events)
    worker._native_exec.get_web_favicon.assert_not_awaited()


def test_scroll_guard_correlates_warning(worker):
    events = collect(worker, "scroll_web_page", {"url": URL, "page": 1})
    assert events[-1]["status"] == "warning"
    assert all(event["tool_call_id"] == "call_1" for event in events)
    worker._native_exec.get_web_favicon.assert_not_awaited()
    worker._native_exec.scroll_url.assert_not_awaited()
    assert worker._native_exec.submit_tool_output.call_args.kwargs["is_error"] is True


def test_external_error_correlates_without_icon(worker):
    worker._native_exec.search_url.return_value = "❌ Keyword not found."
    events = collect(worker, "search_web_page", {"url": URL, "query": "missing"})
    assert events[-1]["status"] == "warning"
    assert all(event["tool_call_id"] == "call_1" for event in events)
    worker._native_exec.get_web_favicon.assert_not_awaited()


def test_validation_failure_correlates_error(worker, monkeypatch):
    monkeypatch.setattr(
        web,
        "ToolValidator",
        lambda: SimpleNamespace(validate_args=lambda *args: "missing arguments"),
    )
    events = collect(worker, "read_web_page", {})
    assert events[-1]["status"] == "error"
    assert all(event["tool_call_id"] == "call_1" for event in events)
    worker._native_exec.create_action.assert_not_awaited()


@pytest.mark.parametrize(
    "url",
    [None, "file:///tmp/icon", "https://user:pass@example.org/", "https://[bad"],
)
def test_unsafe_display_urls_omitted(url):
    assert web._web_status_url(url) is None
