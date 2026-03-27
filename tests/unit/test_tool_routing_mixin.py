"""
tests/test_tool_routing_mixin.py
─────────────────────────────────
Unit tests for ToolRoutingMixin.parse_and_set_function_calls()

Tests the <fc> tag parser, loose fallback, plan tag stripping,
argument normalisation, ID generation, and the JsonUtilsMixin guard.

No DB, Redis, or network required.
"""

from __future__ import annotations

import json

import pytest

from src.api.entities_api.orchestration.mixins.json_utils_mixin import JsonUtilsMixin
from src.api.entities_api.orchestration.mixins.tool_routing_mixin import (
    ToolRoutingMixin,
)

# ---------------------------------------------------------------------------
# Minimal concrete class — satisfies both mixin requirements
# ---------------------------------------------------------------------------


class ConcreteRouter(ToolRoutingMixin, JsonUtilsMixin):
    """Minimal composition for testing — no DB, no Redis, no services."""

    pass


@pytest.fixture
def router():
    return ConcreteRouter()


# ---------------------------------------------------------------------------
# Happy path — <fc> tag parsing
# ---------------------------------------------------------------------------


def test_single_fc_tag_parsed(router):
    payload = json.dumps({"name": "perform_web_search", "arguments": {"query": "test"}})
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert len(result) == 1
    assert result[0]["name"] == "perform_web_search"
    assert result[0]["arguments"]["query"] == "test"


def test_batch_fc_tags_parsed(router):
    p1 = json.dumps({"name": "read_scratchpad", "arguments": {}})
    p2 = json.dumps({"name": "perform_web_search", "arguments": {"query": "ai news"}})
    accumulated = f"<fc>{p1}</fc> some text <fc>{p2}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert len(result) == 2
    names = [r["name"] for r in result]
    assert "read_scratchpad" in names
    assert "perform_web_search" in names


def test_fc_tag_case_insensitive(router):
    payload = json.dumps(
        {"name": "code_interpreter", "arguments": {"code": "print(1)"}}
    )
    accumulated = f"<FC>{payload}</FC>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert len(result) == 1
    assert result[0]["name"] == "code_interpreter"


# ---------------------------------------------------------------------------
# Plan tag stripping — tool calls inside <plan> must be ignored
# ---------------------------------------------------------------------------


def test_fc_inside_plan_tag_is_ignored(router):
    real_payload = json.dumps(
        {"name": "perform_web_search", "arguments": {"query": "real"}}
    )
    plan_payload = json.dumps({"name": "fake_tool", "arguments": {}})
    accumulated = f"<plan><fc>{plan_payload}</fc></plan><fc>{real_payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    names = [r["name"] for r in result]
    assert "fake_tool" not in names
    assert "perform_web_search" in names


# ---------------------------------------------------------------------------
# ID generation — missing IDs are auto-generated
# ---------------------------------------------------------------------------


def test_missing_id_is_generated(router):
    payload = json.dumps({"name": "append_scratchpad", "arguments": {"entry": "test"}})
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert result[0].get("id") is not None
    assert result[0]["id"].startswith("call_")


def test_existing_id_is_preserved(router):
    payload = json.dumps(
        {
            "id": "call_abc123",
            "name": "append_scratchpad",
            "arguments": {"entry": "test"},
        }
    )
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert result[0]["id"] == "call_abc123"


# ---------------------------------------------------------------------------
# Argument normalisation — string arguments are parsed to dict
# ---------------------------------------------------------------------------


def test_string_arguments_normalised_to_dict(router):
    inner_args = json.dumps({"query": "hello world"})
    payload = json.dumps({"name": "perform_web_search", "arguments": inner_args})
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert isinstance(result[0]["arguments"], dict)
    assert result[0]["arguments"]["query"] == "hello world"


def test_markdown_fenced_arguments_normalised(router):
    inner_args = '```json\n{"query": "fenced"}\n```'
    payload = json.dumps({"name": "perform_web_search", "arguments": inner_args})
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert isinstance(result[0]["arguments"], dict)
    assert result[0]["arguments"]["query"] == "fenced"


# ---------------------------------------------------------------------------
# State management after parsing
# ---------------------------------------------------------------------------


def test_tool_response_state_set_true_after_parse(router):
    payload = json.dumps({"name": "code_interpreter", "arguments": {"code": "1+1"}})
    accumulated = f"<fc>{payload}</fc>"
    router.parse_and_set_function_calls(accumulated, "")
    assert router.get_tool_response_state() is True


def test_function_call_state_matches_result(router):
    payload = json.dumps({"name": "file_search", "arguments": {"query": "doc"}})
    accumulated = f"<fc>{payload}</fc>"
    result = router.parse_and_set_function_calls(accumulated, "")
    assert router.get_function_call_state() == result


def test_empty_content_returns_empty_list(router):
    result = router.parse_and_set_function_calls("", "")
    assert result == []
    assert router.get_tool_response_state() is False


def test_set_function_call_state_reset(router):
    router.set_function_call_state([{"name": "tool_a"}])
    assert len(router.get_function_call_state()) == 1
    router.set_function_call_state(None)
    assert router.get_function_call_state() == []


def test_set_function_call_state_single_dict_wrapped(router):
    router.set_function_call_state({"name": "tool_a"})
    assert router.get_function_call_state() == [{"name": "tool_a"}]


# ---------------------------------------------------------------------------
# JsonUtilsMixin guard
# ---------------------------------------------------------------------------


def test_raises_type_error_without_json_utils_mixin():
    """ToolRoutingMixin without JsonUtilsMixin must raise TypeError."""

    class BareRouter(ToolRoutingMixin):
        pass

    bare = BareRouter()
    with pytest.raises(
        TypeError, match="ToolRoutingMixin must be mixed with JsonUtilsMixin"
    ):
        bare.parse_and_set_function_calls("<fc>{}</fc>", "")


# ---------------------------------------------------------------------------
# tools_called tracking
# ---------------------------------------------------------------------------


def test_reset_tools_called(router):
    router._tools_called = ["tool_a", "tool_b"]
    router.reset_tools_called()
    assert router.get_tools_called() == []


def test_get_tools_called_returns_copy(router):
    router._tools_called = ["tool_a"]
    result = router.get_tools_called()
    result.append("tool_b")
    assert router._tools_called == ["tool_a"]
