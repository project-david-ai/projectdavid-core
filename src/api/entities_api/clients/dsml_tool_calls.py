"""Fail-closed compatibility for text-form DeepSeek DSML tool calls.

Converts complete, validated DSML invocations to the existing
<fc> JSON ABI. Does not execute or authorise tools.
"""

from __future__ import annotations

import json
import math
import re

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

_DSML_TAG = re.compile(
    r"<(?P<close>/)?(?P<bar>[?\uFF5C|])DSML(?P=bar)\s*"
    r"(?P<tag>calls|invoke|parameter)\b(?P<attrs>[^>]*)>",
    re.IGNORECASE | re.DOTALL,
)

_ENVELOPE_START = re.compile(r"<fc>|" + _DSML_TAG.pattern, _DSML_TAG.flags)
_FC_TAG = re.compile(r"</?fc>", re.IGNORECASE)
_FC_JSON_START = re.compile(r"\s*(?=\{)")
_FC_CLOSE = re.compile(r"\s*</fc>", re.IGNORECASE)
_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.:\-]{0,127}\Z")

_BARE_AMPERSAND = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#[0-9]+;|#x[0-9A-Fa-f]+;)")

_MAX_FRAGMENT = 256 * 1024
_MAX_CALLS = 16
_MAX_PARAMETERS = 128

# G87_MALFORMED_SNAPSHOT_READONLY_V3
# Only recover the observed malformed, zero-argument, read-only snapshot call.
# The original strict DSML parser retains exclusive authority for all else.
_G87_SNAPSHOT_PATTERN = (
    r"<[｜|]DSML[｜|]\s*calls\s*>\s*"
    r'<[｜|]DSML[｜|]\s*invoke\s+name="playwright-primary__browser_snapshot"\s*>\s*'
    r'<[｜|]DSML[｜|]\s*parameter\s+name="arguments"\s*:\s*\{\s*\}\s*'
    r"</[｜|]DSML[｜|]\s*parameter\s*>\s*"
    r"</[｜|]DSML[｜|]\s*invoke\s*>\s*"
    r"</[｜|]DSML[｜|]\s*calls\s*>"
)
_G87_BAD_SNAPSHOT = re.compile(_G87_SNAPSHOT_PATTERN, re.IGNORECASE | re.DOTALL)
_G87_CANONICAL = (
    '<fc>{"name":"playwright-primary__browser_snapshot","arguments":{}}</fc>'
)


def _whitespace_only(value: str | None) -> bool:
    return not value or not value.strip()


def _valid_name(value: str | None) -> bool:
    return isinstance(value, str) and _TOOL_NAME.fullmatch(value) is not None


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Tool arguments require finite numbers")
    return number


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate argument key")
        result[key] = value
    return result


def _load_arguments(value: str):
    return json.loads(
        value,
        parse_float=_finite_float,
        parse_constant=_finite_float,
        object_pairs_hook=_unique_object,
    )


def _parameters(node) -> dict | None:
    if not _whitespace_only(node.text) or len(node) > _MAX_PARAMETERS:
        return None

    arguments = {}

    for child in node:
        if (
            child.tag != "parameter"
            or not _whitespace_only(child.tail)
            or len(child) != 0
            or not set(child.attrib).issubset({"name", "string"})
        ):
            return None

        key = child.get("name")

        if not _valid_name(key) or key in arguments:
            return None

        flag = child.get("string", "true").casefold()
        value = child.text or ""

        if flag == "true":
            arguments[key] = value
        elif flag == "false":
            try:
                arguments[key] = _load_arguments(value)
            except (ValueError, RecursionError):
                return None
        else:
            return None

    return arguments


def _invocation(node, *, grouped: bool) -> dict | None:
    if node.tag not in {"invoke", "parameter"}:
        return None

    if not set(node.attrib).issubset({"name", "string"}):
        return None

    tool = node.get("name")

    if not _valid_name(tool):
        return None

    if len(node):
        if node.tag == "parameter" and node.get("string", "true") != "true":
            return None

        arguments = _parameters(node)

    elif node.tag == "invoke" and _whitespace_only(node.text):
        arguments = {}

    elif node.tag == "parameter":
        try:
            arguments = _load_arguments(node.text or "")
        except (ValueError, RecursionError):
            return None

        if not isinstance(arguments, dict):
            return None

    else:
        return None

    if arguments is None:
        return None

    # Some providers wrap the complete argument object in a single parameter.
    if (
        grouped
        and node.tag == "invoke"
        and set(node.attrib) == {"name"}
        and len(node) == 1
        and node[0].attrib == {"name": "arguments", "string": "false"}
        and isinstance(arguments.get("arguments"), dict)
    ):
        arguments = arguments["arguments"]

    return {"name": tool, "arguments": arguments}


def _convert_fragment(fragment: str) -> str | None:
    """Convert an entire valid block or reject it atomically."""

    if not fragment or len(fragment) > _MAX_FRAGMENT or not _DSML_TAG.search(fragment):
        return None

    if _G87_BAD_SNAPSHOT.fullmatch(fragment.strip()):
        return _G87_CANONICAL

    def replace_tag(match: re.Match) -> str:
        closing = "/" if match.group("close") else ""
        attrs = match.group("attrs")
        if closing and attrs.strip():
            raise ValueError("Closing tag has attributes")
        if closing:
            attrs = ""
        return f'<{closing}{match.group("tag").lower()}{attrs}>'

    try:
        converted = _DSML_TAG.sub(replace_tag, fragment)
        converted = _BARE_AMPERSAND.sub("&amp;", converted)
        root = fromstring("<root>" + converted + "</root>", forbid_dtd=True)
    except (DefusedXmlException, ParseError, ValueError):
        return None

    if not _whitespace_only(root.text):
        return None

    invocations = []

    for element in root:
        if not _whitespace_only(element.tail):
            return None

        if element.tag == "calls":
            if element.attrib or not _whitespace_only(element.text):
                return None

            invocations.extend((child, True) for child in element)

            if any(not _whitespace_only(child.tail) for child in element):
                return None
        else:
            invocations.append((element, False))

    if not 1 <= len(invocations) <= _MAX_CALLS:
        return None

    calls = []

    for node, grouped in invocations:
        call = _invocation(node, grouped=grouped)

        if call is None:
            return None

        calls.append(call)

    try:
        payloads = [
            json.dumps(call, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            for call in calls
        ]
    except (ValueError, RecursionError):
        return None

    # A literal angle bracket in a JSON string must not become an ABI delimiter.
    return "\n".join(
        "<fc>" + payload.replace("<", r"\u003c") + "</fc>" for payload in payloads
    )


def _dsml_span(text: str, offset: int) -> int | None:
    """Find the end of one balanced DSML outer element."""

    stack = []

    for match in _DSML_TAG.finditer(text, offset):
        if match.start() == offset and match.group("close"):
            return None

        if match.group("close"):
            if not stack or stack.pop() != match.group("tag").lower():
                return None

            if not stack:
                return match.end()

        else:
            stack.append(match.group("tag").lower())

    return None


def _fc_span(text: str, start: int) -> tuple[int, str | None] | None:
    """Find a complete fc envelope. An existing JSON call needs no conversion."""
    body_start = start + len("<fc>")
    json_start = _FC_JSON_START.match(text, body_start)
    if json_start:
        try:
            _, end = json.JSONDecoder().raw_decode(text, json_start.end())
        except (ValueError, RecursionError):
            pass
        else:
            closing = _FC_CLOSE.match(text, end)
            if closing:
                return closing.end(), None

    depth = 1
    for tag in _FC_TAG.finditer(text, body_start):
        depth += 1 if tag.group().lower() == "<fc>" else -1
        if depth == 0:
            return tag.end(), text[body_start : tag.start()]
    return None


def normalise_dsml_calls(text: str) -> str:
    """Translate complete outer DSML envelopes into Project David tool calls."""
    if not isinstance(text, str) or not _DSML_TAG.search(text):
        return text

    output = []
    scan_from = emit_from = 0
    while match := _ENVELOPE_START.search(text, scan_from):
        start = match.start()
        if match.group().lower() == "<fc>":
            span = _fc_span(text, start)
            if span is None:
                break
            end, fragment = span
        else:
            # The observed malformed snapshot is allowed only as an entire call.
            snapshot = _G87_BAD_SNAPSHOT.match(text, start)
            end = snapshot.end() if snapshot else _dsml_span(text, start)
            if end is None:
                break
            fragment = text[start:end]

        scan_from = end
        replacement = _convert_fragment(fragment) if fragment is not None else None
        if replacement is not None:
            output.extend((text[emit_from:start], replacement))
            emit_from = end

    output.append(text[emit_from:])
    return "".join(output)
