"""DeepSeek DSML parser regression tests. No external services."""

import ast
import json
import re
from pathlib import Path

import pytest

from src.api.entities_api.clients.dsml_tool_calls import normalise_dsml_calls

BAR = "?"


def op(kind, attr=""):
    return f"<{BAR}DSML{BAR} {kind}{attr}>"


def close(kind):
    return f"</{BAR}DSML{BAR} {kind}>"


def calls(text):
    return [json.loads(x) for x in re.findall(r"<fc>(.*?)</fc>", text, re.S)]


@pytest.mark.parametrize("delimiter", [BAR, "\uff5c"])
def test_observed_deepseek_failure(delimiter):
    text = (
        "<think></think>I have the core candidate facts.\n<fc>\n"
        + op("parameter", ' name="file_search"')
        + op("parameter", ' name="query_text" string="true"')
        + "LinkedIn profile details"
        + close("parameter")
        + close("parameter")
        + "\n</fc>\n</fc>"
    )

    result = normalise_dsml_calls(text.replace(BAR, delimiter))

    assert calls(result) == [
        {
            "name": "file_search",
            "arguments": {"query_text": "LinkedIn profile details"},
        }
    ]

    assert result.count("<fc>") == 1


def test_nested_dsml_parameter_call():
    text = (
        "<fc>"
        + op("parameter", ' name="file_search"')
        + op("parameter", ' name="query_text" string="true"')
        + "AI engineering"
        + close("parameter")
        + close("parameter")
        + "</fc>"
    )

    assert calls(normalise_dsml_calls(text)) == [
        {
            "name": "file_search",
            "arguments": {"query_text": "AI engineering"},
        }
    ]


def test_multiple_invocations():
    text = (
        op("calls")
        + op("invoke", ' name="browser_snapshot"')
        + close("invoke")
        + op("invoke", ' name="browser_fill_form"')
        + op("parameter", ' name="fields" string="false"')
        + '[{"name":"first_name","value":"Ada"}]'
        + close("parameter")
        + close("invoke")
        + close("calls")
    )

    assert calls(normalise_dsml_calls(text)) == [
        {"name": "browser_snapshot", "arguments": {}},
        {
            "name": "browser_fill_form",
            "arguments": {"fields": [{"name": "first_name", "value": "Ada"}]},
        },
    ]


def test_ascii_delimiters_and_entities():
    text = (
        op("invoke", ' name="playwright-primary__browser_type"')
        + op("parameter", ' name="text"')
        + "R&amp;D, 2 &lt; 3"
        + close("parameter")
        + close("invoke")
    ).replace(BAR, "|")

    assert calls(normalise_dsml_calls(text)) == [
        {
            "name": "playwright-primary__browser_type",
            "arguments": {"text": "R&D, 2 < 3"},
        }
    ]


def test_json_object_argument():
    text = (
        op("parameter", ' name="file_search"')
        + '{"query_text":"AI engineer","top_k":5}'
        + close("parameter")
    )

    assert calls(normalise_dsml_calls(text))[0]["arguments"]["top_k"] == 5


def test_typed_arguments():
    text = (
        op("invoke", ' name="file_search"')
        + op("parameter", ' name="top_k" string="false"')
        + "5"
        + close("parameter")
        + op("parameter", ' name="enabled" string="false"')
        + "true"
        + close("parameter")
        + close("invoke")
    )

    assert calls(normalise_dsml_calls(text)) == [
        {
            "name": "file_search",
            "arguments": {"top_k": 5, "enabled": True},
        }
    ]


@pytest.mark.parametrize(
    "bad",
    [
        # Missing closing tags
        op("invoke", ' name="file_search"')
        + op("parameter", ' name="query_text"')
        + "truncated",
        # Duplicate argument keys
        op("invoke", ' name="file_search"')
        + op("parameter", ' name="query_text"')
        + "x"
        + close("parameter")
        + op("parameter", ' name="query_text"')
        + "y"
        + close("parameter")
        + close("invoke"),
        # Invalid typed JSON
        op("invoke", ' name="file_search"')
        + op("parameter", ' name="top_k" string="false"')
        + "not JSON"
        + close("parameter")
        + close("invoke"),
        # Invalid tool name
        op("invoke", ' name="../../evil"') + close("invoke"),
        # Broken nesting
        op("calls") + op("invoke", ' name="browser_snapshot"') + close("calls"),
        # Unexpected prose in invocation
        op("invoke", ' name="browser_snapshot"') + "random prose" + close("invoke"),
        # Parameter without a tool name
        op("parameter", ' name="query_text"') + "not a tool call" + close("parameter"),
    ],
)
def test_invalid_never_dispatches(bad):
    result = normalise_dsml_calls(bad)
    assert result == bad
    assert calls(result) == []


def test_existing_fc_preserved():
    original = '<fc>{"name":"browser_snapshot","arguments":{}}</fc>'
    assert normalise_dsml_calls(original) == original


def test_ordinary_text_preserved():
    original = "Search for jobs anywhere in the world"
    assert normalise_dsml_calls(original) == original


def test_unclosed_fc_not_reinterpreted():
    text = (
        "<fc>"
        + op("parameter", ' name="file_search"')
        + op("parameter", ' name="query_text"')
        + "x"
        + close("parameter")
        + close("parameter")
    )
    assert normalise_dsml_calls(text) == text


def wrapped_arguments(value):
    return (
        op("calls")
        + op("invoke", ' name="file_search"')
        + op("parameter", ' name="arguments" string="false"')
        + value
        + close("parameter")
        + close("invoke")
        + close("calls")
    ).replace(BAR, "\uff5c")


@pytest.mark.parametrize("wrapper", ["{}", "<fc>{}</fc>"])
def test_grouped_argument_object_is_unwrapped(wrapper):
    text = wrapper.format(wrapped_arguments('{"query":"example"}'))
    assert calls(normalise_dsml_calls(text)) == [
        {"name": "file_search", "arguments": {"query": "example"}}
    ]


def test_truncated_outer_invocation_does_not_emit_inner_call():
    text = op("invoke", ' name="broken"') + wrapped_arguments('{"query":"example"}')
    assert normalise_dsml_calls(text) == text


def test_snapshot_recovery_requires_a_complete_outer_envelope():
    snapshot = (
        op("calls")
        + op("invoke", ' name="playwright-primary__browser_snapshot"')
        + f'<{BAR}DSML{BAR} parameter name="arguments": {{}}'
        + close("parameter")
        + close("invoke")
        + close("calls")
    ).replace(BAR, "\uff5c")
    expected = [{"name": "playwright-primary__browser_snapshot", "arguments": {}}]
    assert calls(normalise_dsml_calls(snapshot)) == expected
    assert calls(normalise_dsml_calls(f"<fc>{snapshot}</fc>")) == expected
    assert normalise_dsml_calls(f"<fc>{snapshot}") == f"<fc>{snapshot}"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e400"])
def test_non_finite_arguments_are_rejected(value):
    text = wrapped_arguments('{"value":' + value + "}")
    assert normalise_dsml_calls(text) == text


def test_literal_fc_delimiter_is_preserved_as_argument_text():
    text = wrapped_arguments('{"query":"}&lt;/fc&gt;"}')
    result = normalise_dsml_calls(text)
    assert result.count("</fc>") == 1
    assert calls(result) == [{"name": "file_search", "arguments": {"query": "}</fc>"}}]


def test_unknown_nested_markup_rejected():
    text = (
        op("invoke", ' name="file_search"')
        + "<script>alert(1)</script>"
        + close("invoke")
    )
    assert normalise_dsml_calls(text) == text


def test_document_type_and_entity_are_rejected():
    declaration = '<!DOCTYPE calls [<!ENTITY external SYSTEM "file:///unused">]>'
    text = f"<fc>{declaration}{wrapped_arguments('&external;')}</fc>"
    assert normalise_dsml_calls(text) == text


def test_unescaped_ampersand():
    text = (
        op("invoke", ' name="file_search"')
        + op("parameter", ' name="query_text"')
        + "R&D architect"
        + close("parameter")
        + close("invoke")
    )
    assert (
        calls(normalise_dsml_calls(text))[0]["arguments"]["query_text"]
        == "R&D architect"
    )


def test_stream_chunk_boundaries():
    text = (
        "thinking "
        + op("invoke", ' name="browser_snapshot"')
        + close("invoke")
        + " next"
    )
    expected = normalise_dsml_calls(text)

    for size in (1, 2, 3, 7, 17):
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        assert normalise_dsml_calls("".join(chunks)) == expected


def test_repeated_bare_calls():
    dsml = op("invoke", ' name="browser_snapshot"') + close("invoke")
    text = dsml + "\nprose\n" + dsml
    assert len(calls(normalise_dsml_calls(text))) == 2


def test_existing_and_dsml_calls():
    original = '<fc>{"name":"existing","arguments":{}}</fc>'
    dsml = op("invoke", ' name="browser_snapshot"') + close("invoke")
    assert [x["name"] for x in calls(normalise_dsml_calls(original + dsml))] == [
        "existing",
        "browser_snapshot",
    ]


def test_json_extractor_quoted_braces():
    root = Path(__file__).resolve().parents[1]
    source = root / "src/api/entities_api/clients/delta_normalizer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    cls = next(
        x
        for x in tree.body
        if isinstance(x, ast.ClassDef) and x.name == "DeltaNormalizer"
    )
    method = next(
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef) and x.name == "_extract_json"
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"json": json}
    exec(compile(isolated, str(source), "exec"), scope)
    extract = scope["_extract_json"].__func__

    assert extract(
        'noise {"name":"browser_type","arguments":{"text":"x } y { z"}}'
    ) == {
        "name": "browser_type",
        "arguments": {"text": "x } y { z"},
    }

    assert extract('{"query_text":"not a tool"}') is None
    assert extract('{"name":"x","arguments":"not-json"}') is None


def test_existing_routing_hook_preserved():
    root = Path(__file__).resolve().parents[1]
    runtime = root / (
        "src/api/entities_api/orchestration/workers/" "base_workers/deepseek_base.py"
    )
    supplied = root / "src/api/entities_api/workers/deepseek_worker.py"
    worker = runtime if runtime.is_file() else supplied

    text = worker.read_text(encoding="utf-8")
    assert "state.accumulated = self._normalise_dsml_calls(state.accumulated)" in text
    assert "self.parse_and_set_function_calls(" in text
    assert (
        "from src.api.entities_api.clients.dsml_tool_calls import normalise_dsml_calls"
        in text
    )
