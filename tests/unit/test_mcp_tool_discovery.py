"""MCP-2 tests for MCP tool discovery and provider-safe identities."""

from __future__ import annotations

import re

import pytest
from mcp.types import ListToolsResult, Tool

from src.api.entities_api.orchestration.mcp_tool_discovery import (
    adapt_mcp_list_tools_result,
    adapt_mcp_tools,
)

_PROVIDER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tool(
    name: str,
    *,
    description: str | None = None,
    title: str | None = None,
    input_schema: dict | None = None,
) -> Tool:
    payload = {
        "name": name,
        "inputSchema": input_schema
        or {
            "type": "object",
            "properties": {},
        },
    }

    if description is not None:
        payload["description"] = description

    if title is not None:
        payload["title"] = title

    return Tool.model_validate(payload)


def test_maps_mcp_tool_to_namespaced_identity_and_tool_definition():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    (tool,) = adapt_mcp_tools(
        "github",
        [_tool("search_issues", description="Search issues", input_schema=schema)],
    )

    assert tool.server_id == "github"
    assert tool.remote_name == "search_issues"
    assert tool.canonical_id == "mcp:github:search_issues"
    assert tool.provider_name == "github__search_issues"

    assert tool.definition.name == "github__search_issues"
    assert tool.definition.description == "Search issues"
    assert tool.definition.input_schema == schema

    assert tool.definition.to_function_tool() == {
        "type": "function",
        "function": {
            "name": "github__search_issues",
            "description": "Search issues",
            "parameters": schema,
        },
    }


def test_title_is_used_when_mcp_description_is_absent():
    (tool,) = adapt_mcp_tools(
        "drive",
        [_tool("search", title="Search Google Drive")],
    )

    assert tool.definition.description == "Search Google Drive"


def test_provider_name_normalizes_mcp_characters_not_allowed_by_provider_shape():
    (tool,) = adapt_mcp_tools(
        "google.drive",
        [_tool("files.read")],
    )

    assert tool.provider_name == "google_drive__files_read"
    assert _PROVIDER_NAME.fullmatch(tool.provider_name)


def test_normalization_collisions_receive_distinct_deterministic_aliases():
    first = adapt_mcp_tools(
        "github",
        [_tool("read.file"), _tool("read_file")],
    )

    second = adapt_mcp_tools(
        "github",
        [_tool("read_file"), _tool("read.file")],
    )

    first_by_remote = {tool.remote_name: tool.provider_name for tool in first}
    second_by_remote = {tool.remote_name: tool.provider_name for tool in second}

    assert first_by_remote == second_by_remote
    assert first_by_remote["read.file"] != first_by_remote["read_file"]

    for provider_name in first_by_remote.values():
        assert _PROVIDER_NAME.fullmatch(provider_name)
        assert len(provider_name) <= 64


def test_reserved_provider_name_is_disambiguated():
    (tool,) = adapt_mcp_tools(
        "github",
        [_tool("search_issues")],
        reserved_provider_names={"github__search_issues"},
    )

    assert tool.provider_name != "github__search_issues"
    assert tool.provider_name.startswith("github__search_issues")
    assert _PROVIDER_NAME.fullmatch(tool.provider_name)


def test_long_provider_name_is_stably_truncated_with_hash():
    remote_name = "search_" + ("very_long_component_" * 8)

    (first,) = adapt_mcp_tools("enterprise-production", [_tool(remote_name)])
    (second,) = adapt_mcp_tools("enterprise-production", [_tool(remote_name)])

    assert first.provider_name == second.provider_name
    assert len(first.provider_name) <= 64
    assert _PROVIDER_NAME.fullmatch(first.provider_name)


def test_canonical_identity_escapes_namespace_separators_without_losing_remote_name():
    (tool,) = adapt_mcp_tools(
        "tenant:prod",
        [_tool("files:read")],
    )

    assert tool.remote_name == "files:read"
    assert tool.canonical_id == "mcp:tenant%3Aprod:files%3Aread"


def test_duplicate_remote_tool_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate tool names"):
        adapt_mcp_tools(
            "github",
            [_tool("search"), _tool("search")],
        )


def test_blank_server_id_is_rejected():
    with pytest.raises(ValueError, match="server_id"):
        adapt_mcp_tools("   ", [_tool("search")])


def test_input_schema_is_copied_out_of_sdk_model():
    remote = _tool(
        "search",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )

    (tool,) = adapt_mcp_tools("github", [remote])

    remote.input_schema["properties"]["query"]["type"] = "integer"

    assert tool.definition.input_schema["properties"]["query"]["type"] == "string"


def test_list_tools_page_preserves_cursor_and_adapts_tools():
    result = ListToolsResult.model_validate(
        {
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
            "nextCursor": "cursor_2",
        }
    )

    page = adapt_mcp_list_tools_result("github", result)

    assert page.next_cursor == "cursor_2"
    assert len(page.tools) == 1
    assert page.tools[0].canonical_id == "mcp:github:search"
