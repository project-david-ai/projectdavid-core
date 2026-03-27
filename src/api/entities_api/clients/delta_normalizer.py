# src/api/entities_api/clients/delta_normalizer.py
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from projectdavid_common.utilities.logging_service import LoggingUtility

load_dotenv()
LOG = LoggingUtility()

# ── C accelerator (optional) ──────────────────────────────────────────────────
# If the extension is built and importable, the inner `while buffer:` loop is
# replaced by a single C call that processes the entire buffer in one shot.
# Falls back transparently to the pure-Python implementation if unavailable.
try:
    import delta_normalizer_core as _C

    _C_AVAILABLE = True
    LOG.info("DeltaNormalizer: C accelerator loaded (delta_normalizer_core)")

    # Map Python state-string names → C int constants
    _PY_STATE_TO_INT: dict[str, int] = {
        "content": _C.ST_CONTENT,
        "think": _C.ST_THINK,
        "plan": _C.ST_PLAN,
        "decision": _C.ST_DECISION,
        "fc": _C.ST_FC,
        "tool_call_xml": _C.ST_TOOL_CALL_XML,
        "tool_code_xml": _C.ST_TOOL_CODE_XML,
        "md_json_block": _C.ST_MD_JSON_BLOCK,
        "naked_json": _C.ST_NAKED_JSON,
        "kimi_router": _C.ST_KIMI_ROUTER,
        "kimi_args": _C.ST_KIMI_ARGS,
        "unicode_tool_router": _C.ST_UNICODE_TOOL_ROUTER,
        "unicode_tool_parsing": _C.ST_UNICODE_TOOL_PARSING,
        "unicode_tool_args": _C.ST_UNICODE_TOOL_ARGS,
        "channel_reasoning": _C.ST_CHANNEL_REASONING,
        "channel_tool_meta": _C.ST_CHANNEL_TOOL_META,
        "channel_tool_payload": _C.ST_CHANNEL_TOOL_PAYLOAD,
    }
    _INT_TO_PY_STATE: dict[int, str] = {v: k for k, v in _PY_STATE_TO_INT.items()}

except ImportError:
    _C_AVAILABLE = False
    LOG.warning(
        "DeltaNormalizer: C accelerator not found — running pure Python. "
        "Build with:  cd c_extensions && python setup.py build_ext --inplace"
    )


class DeltaNormalizer:
    FC_START, FC_END = "<fc>", "</fc>"
    TC_START, TC_END = "<tool_call>", "</tool_call>"
    TCODE_START, TCODE_END = "<tool_code>", "</tool_code>"
    MD_JSON_START = "```json"
    MD_END = "```"
    NAKED_JSON_START = "{"
    TH_START, TH_END = "<think>", "</think>"
    DEC_START, DEC_END = "<decision>", "</decision>"
    PLAN_START, PLAN_END = "<plan>", "</plan>"
    CH_ANALYSIS = "<|channel|>analysis"
    CH_COMMENTARY = "<|channel|>commentary"
    CH_FINAL = "<|channel|>final"
    MSG_TAG = "<|message|>"
    CALL_TAG = "<|call|>"
    KIMI_SEC_START = "<|tool_calls_section_begin|>"
    KIMI_SEC_END = "<|tool_calls_section_end|>"
    KIMI_TC_START = "<|tool_call_begin|>"
    KIMI_ARG_START = "<|tool_call_argument_begin|>"
    KIMI_TC_END = "<|tool_call_end|>"
    UNICODE_TC_BEGIN = "<｜tool▁calls▁begin｜>"
    UNICODE_TC_END = "<｜tool▁calls▁end｜>"
    UNICODE_CALL_BEGIN = "<｜tool▁call▁begin｜>"
    UNICODE_CALL_END = "<｜tool▁call▁end｜>"
    UNICODE_SEP = "<｜tool▁sep｜>"

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        Bulletproof JSON extractor.
        Scans messy text (e.g. LLM rambling) to find and parse the first valid
        JSON object.
        """
        start = 0
        while True:
            start = text.find("{", start)
            if start == -1:
                break
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start += 1
        return None

    # ── C-accelerated buffer processor ───────────────────────────────────────
    @classmethod
    def _process_buffer_c(
        cls,
        buffer: str,
        state_str: str,
        json_depth: int,
        has_emitted_text: bool,
        xml_tool_buffer: str,
        run_id: str,
    ):
        """
        Calls the C extension to process `buffer` through the state machine.

        Returns a generator of event dicts, plus updates the mutable state
        variables via the returned tuple:
            (events, new_buffer, new_state_str, new_json_depth,
             new_has_emitted_text, new_xml_tool_buffer)

        NOTE: the C layer emits a special "tool_call_raw_xml" event when it
        finishes an xml-tool block.  We intercept it here and convert it into
        the proper "tool_call" event via _extract_json, keeping JSON parsing
        in Python where it already lives.
        """
        state_int = _PY_STATE_TO_INT.get(state_str, _C.ST_CONTENT)

        events_raw, new_buf, new_state_int, new_depth, new_het, new_xml = (
            _C.process_buffer(
                buffer,
                state_int,
                json_depth,
                int(has_emitted_text),
                xml_tool_buffer,
                run_id,
            )
        )

        # Post-process events: resolve tool_call_raw_xml → tool_call
        events_out = []
        for ev in events_raw:
            if ev["type"] == "tool_call_raw_xml":
                parsed = cls._extract_json(ev["content"])
                if parsed:
                    name = parsed.get("name", "unknown_tool")
                    args = parsed.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    events_out.append(
                        {
                            "type": "tool_call",
                            "content": {"name": name, "arguments": args},
                            "run_id": run_id,
                        }
                    )
                else:
                    LOG.warning(
                        "C: failed to extract JSON from xml buffer. payload: %s",
                        ev["content"],
                    )
            else:
                events_out.append(ev)

        new_state_str = _INT_TO_PY_STATE.get(new_state_int, "content")
        return (
            events_out,
            new_buf,
            new_state_str,
            new_depth,
            bool(new_het),
            new_xml,
        )

    @classmethod
    async def async_iter_deltas(
        cls, raw_stream: AsyncGenerator[Any, None], run_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        buffer = ""
        state = "content"
        json_depth = 0
        has_emitted_text = False
        xml_tool_buffer = ""

        pending_tool_calls: dict = defaultdict(
            lambda: {"index": 0, "function": {"name": "", "arguments": ""}}
        )

        async for token in raw_stream:
            is_dict = isinstance(token, dict)
            delta = {}
            finish_reason = None

            if is_dict:
                if "done" in token or "message" in token or "response" in token:
                    if "message" in token:
                        delta = token.get("message", {})
                    else:
                        delta = {"content": token.get("response", "")}
                    if token.get("done"):
                        finish_reason = token.get("done_reason") or "stop"
                else:
                    choices = token.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")
            else:
                if hasattr(token, "message"):
                    delta = getattr(token, "message", {})
                elif hasattr(token, "response"):
                    delta = {"content": getattr(token, "response", "")}
                if getattr(token, "done", False):
                    finish_reason = getattr(token, "done_reason", "stop")

            # 1. Native API fields (reasoning, tool_calls)
            if isinstance(delta, dict):
                r_content = delta.get("reasoning_content") or delta.get("thinking")
                t_calls = delta.get("tool_calls")
                seg = delta.get("content", "")
            else:
                r_content = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "thinking", None
                )
                t_calls = getattr(delta, "tool_calls", None)
                seg = getattr(delta, "content", "")

            if r_content:
                yield {"type": "reasoning", "content": r_content, "run_id": run_id}

            if t_calls:
                for tc in t_calls:
                    if isinstance(tc, dict):
                        t_index = tc.get("index", 0)
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        fn_args = fn.get("arguments", "")
                    else:
                        t_index = getattr(tc, "index", 0)
                        fn = getattr(tc, "function", None)
                        fn_name = getattr(fn, "name", None) if fn else None
                        fn_args = getattr(fn, "arguments", "") if fn else ""

                    if isinstance(fn_args, dict):
                        fn_args = json.dumps(fn_args)
                    elif fn_args is None:
                        fn_args = ""

                    tool_data = pending_tool_calls[t_index]
                    if fn_name:
                        tool_data["function"]["name"] += fn_name
                        yield {
                            "type": "call_arguments",
                            "content": fn_name,
                            "run_id": run_id,
                        }
                    if fn_args:
                        tool_data["function"]["arguments"] += fn_args
                        yield {
                            "type": "call_arguments",
                            "content": fn_args,
                            "run_id": run_id,
                        }

            seg = seg or ""

            if finish_reason == "tool_calls":
                for idx, data in list(pending_tool_calls.items()):
                    name = data["function"]["name"]
                    args = data["function"]["arguments"]
                    if name:
                        yield {
                            "type": "tool_call",
                            "content": {"name": name, "arguments": args},
                            "run_id": run_id,
                        }
                pending_tool_calls.clear()

            if not seg:
                continue

            buffer += seg

            # 2. Buffer → state machine
            if _C_AVAILABLE:
                events, buffer, state, json_depth, has_emitted_text, xml_tool_buffer = (
                    cls._process_buffer_c(
                        buffer,
                        state,
                        json_depth,
                        has_emitted_text,
                        xml_tool_buffer,
                        run_id,
                    )
                )
                for ev in events:
                    LOG.warning(
                        "C_EVENT: type=%s content=%s",
                        ev.get("type"),
                        str(ev.get("content", ""))[:80],
                    )
                    yield ev
            else:
                # ── Pure-Python fallback ───────────────────────────────────
                while buffer:
                    yielded_something = False

                    if state == "content":
                        if (
                            "<" not in buffer
                            and "`" not in buffer
                            and "{" not in buffer
                        ):
                            if buffer.strip():
                                has_emitted_text = True
                            yield {
                                "type": "content",
                                "content": buffer,
                                "run_id": run_id,
                            }
                            buffer = ""
                            break

                        lt_idx = buffer.find("<")
                        bt_idx = buffer.find("`")
                        cur_idx = buffer.find("{")

                        if has_emitted_text:
                            cur_idx = -1

                        indices = [x for x in [lt_idx, bt_idx, cur_idx] if x != -1]

                        if not indices:
                            if buffer.strip():
                                has_emitted_text = True
                            yield {
                                "type": "content",
                                "content": buffer,
                                "run_id": run_id,
                            }
                            buffer = ""
                            break

                        cutoff = min(indices)

                        if cutoff > 0:
                            text_chunk = buffer[:cutoff]
                            if text_chunk.strip():
                                has_emitted_text = True
                            yield {
                                "type": "content",
                                "content": text_chunk,
                                "run_id": run_id,
                            }
                            buffer = buffer[cutoff:]

                        all_tags = [
                            (cls.CH_ANALYSIS, "channel_reasoning"),
                            (cls.CH_COMMENTARY, "channel_tool_meta"),
                            (cls.CH_FINAL, None),
                            (cls.MSG_TAG, None),
                            (cls.FC_START, "fc"),
                            (cls.TC_START, "tool_call_xml"),
                            (cls.TCODE_START, "tool_code_xml"),
                            (cls.MD_JSON_START, "md_json_block"),
                            (cls.TH_START, "think"),
                            (cls.DEC_START, "decision"),
                            (cls.PLAN_START, "plan"),
                            (cls.KIMI_SEC_START, "kimi_router"),
                            (cls.UNICODE_TC_BEGIN, "unicode_tool_router"),
                        ]

                        if not has_emitted_text:
                            all_tags.append((cls.NAKED_JSON_START, "naked_json"))

                        match_found = False
                        for tag, new_state in all_tags:
                            if buffer.startswith(tag):
                                if new_state:
                                    state = new_state
                                    if state == "naked_json":
                                        json_depth = 1
                                    if state in [
                                        "fc",
                                        "tool_call_xml",
                                        "tool_code_xml",
                                        "md_json_block",
                                    ]:
                                        xml_tool_buffer = ""
                                buffer = buffer[len(tag) :]
                                if tag == cls.NAKED_JSON_START:
                                    yield {
                                        "type": "call_arguments",
                                        "content": tag,
                                        "run_id": run_id,
                                    }
                                yielded_something = True
                                match_found = True
                                break

                        if match_found:
                            continue

                        is_partial = any(tag.startswith(buffer) for tag, _ in all_tags)
                        if is_partial:
                            break
                        else:
                            char = buffer[0]
                            if char.strip():
                                has_emitted_text = True
                            yield {"type": "content", "content": char, "run_id": run_id}
                            buffer = buffer[1:]
                            yielded_something = True

                    elif state in ["think", "plan", "decision"]:
                        if state == "think":
                            end_tag, type_name = cls.TH_END, "reasoning"
                        elif state == "plan":
                            end_tag, type_name = cls.PLAN_END, "plan"
                        else:
                            end_tag, type_name = cls.DEC_END, "decision"

                        if "<" not in buffer:
                            yield {
                                "type": type_name,
                                "content": buffer,
                                "run_id": run_id,
                            }
                            buffer = ""
                            break

                        lt_idx = buffer.find("<")
                        if lt_idx > 0:
                            yield {
                                "type": type_name,
                                "content": buffer[:lt_idx],
                                "run_id": run_id,
                            }
                            buffer = buffer[lt_idx:]

                        if buffer.startswith(end_tag):
                            buffer = buffer[len(end_tag) :]
                            state = "content"
                            yielded_something = True
                            continue

                        if end_tag.startswith(buffer):
                            break

                        yield {
                            "type": type_name,
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state in [
                        "fc",
                        "tool_call_xml",
                        "tool_code_xml",
                        "md_json_block",
                    ]:
                        end_tag = {
                            "fc": cls.FC_END,
                            "tool_call_xml": cls.TC_END,
                            "tool_code_xml": cls.TCODE_END,
                            "md_json_block": cls.MD_END,
                        }[state]

                        if buffer.startswith(end_tag):
                            buffer = buffer[len(end_tag) :]
                            state = "content"
                            yielded_something = True

                            if xml_tool_buffer:
                                parsed = cls._extract_json(xml_tool_buffer)
                                if parsed:
                                    name = parsed.get("name", "unknown_tool")
                                    args = parsed.get("arguments", {})
                                    if isinstance(args, dict):
                                        args = json.dumps(args)
                                    yield {
                                        "type": "tool_call",
                                        "content": {"name": name, "arguments": args},
                                        "run_id": run_id,
                                    }
                                else:
                                    LOG.warning(
                                        "Failed to extract valid JSON from XML buffer. "
                                        "Payload: %s",
                                        xml_tool_buffer,
                                    )
                                xml_tool_buffer = ""
                            continue

                        if end_tag.startswith(buffer):
                            break

                        first_char = end_tag[0]
                        idx = buffer.find(first_char)

                        if idx == -1:
                            xml_tool_buffer += buffer
                            yield {
                                "type": "call_arguments",
                                "content": buffer,
                                "run_id": run_id,
                            }
                            buffer = ""
                            break
                        elif idx > 0:
                            xml_tool_buffer += buffer[:idx]
                            yield {
                                "type": "call_arguments",
                                "content": buffer[:idx],
                                "run_id": run_id,
                            }
                            buffer = buffer[idx:]
                            yielded_something = True
                        else:
                            xml_tool_buffer += buffer[0]
                            yield {
                                "type": "call_arguments",
                                "content": buffer[0],
                                "run_id": run_id,
                            }
                            buffer = buffer[1:]
                            yielded_something = True

                    elif state == "naked_json":
                        if not buffer:
                            break
                        chars_processed = 0
                        complete_json = False
                        for char in buffer:
                            if char == "{":
                                json_depth += 1
                            elif char == "}":
                                json_depth -= 1
                            chars_processed += 1
                            if json_depth == 0:
                                complete_json = True
                                break
                        chunk = buffer[:chars_processed]
                        yield {
                            "type": "call_arguments",
                            "content": chunk,
                            "run_id": run_id,
                        }
                        buffer = buffer[chars_processed:]
                        if complete_json:
                            state = "content"
                            yielded_something = True
                        else:
                            break

                    elif state == "kimi_router":
                        if buffer.startswith(cls.KIMI_SEC_END):
                            buffer = buffer[len(cls.KIMI_SEC_END) :]
                            state = "content"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.KIMI_ARG_START):
                            buffer = buffer[len(cls.KIMI_ARG_START) :]
                            state = "kimi_args"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.KIMI_TC_START):
                            buffer = buffer[len(cls.KIMI_TC_START) :]
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.KIMI_TC_END):
                            buffer = buffer[len(cls.KIMI_TC_END) :]
                            yielded_something = True
                            continue
                        router_tags = [
                            cls.KIMI_SEC_END,
                            cls.KIMI_ARG_START,
                            cls.KIMI_TC_START,
                            cls.KIMI_TC_END,
                        ]
                        if any(tag.startswith(buffer) for tag in router_tags):
                            break
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "kimi_args":
                        if buffer.startswith(cls.KIMI_TC_END):
                            buffer = buffer[len(cls.KIMI_TC_END) :]
                            state = "kimi_router"
                            yielded_something = True
                            continue
                        if cls.KIMI_TC_END.startswith(buffer):
                            break
                        yield {
                            "type": "call_arguments",
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "unicode_tool_router":
                        if buffer.startswith(cls.UNICODE_TC_END):
                            buffer = buffer[len(cls.UNICODE_TC_END) :]
                            state = "content"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.UNICODE_CALL_BEGIN):
                            buffer = buffer[len(cls.UNICODE_CALL_BEGIN) :]
                            state = "unicode_tool_parsing"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.UNICODE_CALL_END):
                            buffer = buffer[len(cls.UNICODE_CALL_END) :]
                            yielded_something = True
                            continue
                        router_tags = [
                            cls.UNICODE_TC_END,
                            cls.UNICODE_CALL_BEGIN,
                            cls.UNICODE_CALL_END,
                        ]
                        if any(tag.startswith(buffer) for tag in router_tags):
                            break
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "unicode_tool_parsing":
                        if buffer.startswith(cls.UNICODE_SEP):
                            buffer = buffer[len(cls.UNICODE_SEP) :]
                            state = "unicode_tool_args"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.UNICODE_CALL_END):
                            buffer = buffer[len(cls.UNICODE_CALL_END) :]
                            state = "unicode_tool_router"
                            yielded_something = True
                            continue
                        check_tags = [cls.UNICODE_SEP, cls.UNICODE_CALL_END]
                        if any(tag.startswith(buffer) for tag in check_tags):
                            break
                        yield {
                            "type": "call_arguments",
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "unicode_tool_args":
                        if buffer.startswith(cls.UNICODE_CALL_END):
                            buffer = buffer[len(cls.UNICODE_CALL_END) :]
                            state = "unicode_tool_router"
                            yielded_something = True
                            continue
                        if cls.UNICODE_CALL_END.startswith(buffer):
                            break
                        yield {
                            "type": "call_arguments",
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "channel_reasoning":
                        if buffer.startswith(cls.CH_FINAL):
                            buffer = buffer[len(cls.CH_FINAL) :]
                            state = "content"
                            yielded_something = True
                            continue
                        if buffer.startswith(cls.CH_COMMENTARY):
                            buffer = buffer[len(cls.CH_COMMENTARY) :]
                            state = "channel_tool_meta"
                            yielded_something = True
                            continue
                        specials = [cls.CH_FINAL, cls.CH_COMMENTARY]
                        if any(tag.startswith(buffer) for tag in specials):
                            break
                        yield {
                            "type": "reasoning",
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    elif state == "channel_tool_meta":
                        if cls.MSG_TAG in buffer:
                            _, post = buffer.split(cls.MSG_TAG, 1)
                            state = "channel_tool_payload"
                            buffer = post
                            yielded_something = True
                        elif cls.CH_FINAL in buffer:
                            state = "content"
                            buffer = ""
                            yielded_something = True
                        else:
                            break

                    elif state == "channel_tool_payload":
                        exit_tags = [cls.CALL_TAG, cls.CH_FINAL, cls.CH_ANALYSIS]
                        if any(buffer.startswith(t) for t in exit_tags):
                            matched_t = next(
                                t for t in exit_tags if buffer.startswith(t)
                            )
                            state = (
                                "channel_reasoning"
                                if matched_t == cls.CH_ANALYSIS
                                else "content"
                            )
                            buffer = buffer[len(matched_t) :]
                            yielded_something = True
                            continue
                        if any(t.startswith(buffer) for t in exit_tags):
                            break
                        yield {
                            "type": "call_arguments",
                            "content": buffer[0],
                            "run_id": run_id,
                        }
                        buffer = buffer[1:]
                        yielded_something = True

                    if not yielded_something:
                        break

        # ── Flush pending tool calls ──────────────────────────────────────────
        if pending_tool_calls:
            for idx, data in list(pending_tool_calls.items()):
                name = data["function"]["name"]
                args = data["function"]["arguments"]
                if name:
                    yield {
                        "type": "tool_call",
                        "content": {"name": name, "arguments": args},
                        "run_id": run_id,
                    }

        # ── Flush residual buffer ─────────────────────────────────────────────
        if buffer:
            if state in ["channel_reasoning", "think"]:
                yield {"type": "reasoning", "content": buffer, "run_id": run_id}
            elif state in [
                "channel_tool_payload",
                "fc",
                "tool_call_xml",
                "tool_code_xml",
                "md_json_block",
                "naked_json",
            ]:
                yield {"type": "call_arguments", "content": buffer, "run_id": run_id}
            elif state == "decision":
                yield {"type": "decision", "content": buffer, "run_id": run_id}
            elif state == "plan":
                yield {"type": "plan", "content": buffer, "run_id": run_id}
            elif state == "content":
                yield {"type": "content", "content": buffer, "run_id": run_id}
