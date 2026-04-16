use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ── State constants (same numeric values as C extension) ──────────────────────
const ST_CONTENT: u8 = 0;
const ST_THINK: u8 = 1;
const ST_PLAN: u8 = 2;
const ST_DECISION: u8 = 3;
const ST_FC: u8 = 4;
const ST_TOOL_CALL_XML: u8 = 5;
const ST_TOOL_CODE_XML: u8 = 6;
const ST_MD_JSON_BLOCK: u8 = 7;
const ST_NAKED_JSON: u8 = 8;
const ST_KIMI_ROUTER: u8 = 9;
const ST_KIMI_ARGS: u8 = 10;
const ST_UNICODE_TOOL_ROUTER: u8 = 11;
const ST_UNICODE_TOOL_PARSING: u8 = 12;
const ST_UNICODE_TOOL_ARGS: u8 = 13;
const ST_CHANNEL_REASONING: u8 = 14;
const ST_CHANNEL_TOOL_META: u8 = 15;
const ST_CHANNEL_TOOL_PAYLOAD: u8 = 16;

// ── Tag constants ─────────────────────────────────────────────────────────────
const FC_START: &str = "<fc>";
const FC_END: &str = "</fc>";
const TC_START: &str = "<tool_call>";
const TC_END: &str = "</tool_call>";
const TCODE_START: &str = "<tool_code>";
const TCODE_END: &str = "</tool_code>";
const MD_JSON_START: &str = "```json";
const MD_END: &str = "```";
const TH_START: &str = "<think>";
const TH_END: &str = "</think>";
const DEC_START: &str = "<decision>";
const DEC_END: &str = "</decision>";
const PLAN_START: &str = "<plan>";
const PLAN_END: &str = "</plan>";
const CH_ANALYSIS: &str = "<|channel|>analysis";
const CH_COMMENTARY: &str = "<|channel|>commentary";
const CH_FINAL: &str = "<|channel|>final";
const MSG_TAG: &str = "<|message|>";
const CALL_TAG: &str = "<|call|>";
const KIMI_SEC_START: &str = "<|tool_calls_section_begin|>";
const KIMI_SEC_END: &str = "<|tool_calls_section_end|>";
const KIMI_TC_START: &str = "<|tool_call_begin|>";
const KIMI_ARG_START: &str = "<|tool_call_argument_begin|>";
const KIMI_TC_END: &str = "<|tool_call_end|>";
const UNICODE_TC_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const UNICODE_TC_END: &str = "<｜tool▁calls▁end｜>";
const UNICODE_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const UNICODE_CALL_END: &str = "<｜tool▁call▁end｜>";
const UNICODE_SEP: &str = "<｜tool▁sep｜>";

// Ordered tag table for the content state.
// Longest/most-specific tags MUST come before shorter ones that share a prefix.
const CONTENT_TAG_TABLE: &[(&str, Option<u8>)] = &[
    (CH_ANALYSIS, Some(ST_CHANNEL_REASONING)),
    (CH_COMMENTARY, Some(ST_CHANNEL_TOOL_META)),
    (CH_FINAL, None),
    (MSG_TAG, None),
    (FC_START, Some(ST_FC)),
    (TC_START, Some(ST_TOOL_CALL_XML)),
    (TCODE_START, Some(ST_TOOL_CODE_XML)),
    (MD_JSON_START, Some(ST_MD_JSON_BLOCK)),
    (TH_START, Some(ST_THINK)),
    (DEC_START, Some(ST_DECISION)),
    (PLAN_START, Some(ST_PLAN)),
    (KIMI_SEC_START, Some(ST_KIMI_ROUTER)),
    (UNICODE_TC_BEGIN, Some(ST_UNICODE_TOOL_ROUTER)),
];

// ── Event ─────────────────────────────────────────────────────────────────────
struct Event {
    ev_type: &'static str,
    content: String,
    run_id: String,
}

// ── Processing context ────────────────────────────────────────────────────────
struct Ctx {
    buf: String,
    state: u8,
    json_depth: i32,
    het: bool,   // has_emitted_text
    xtb: String, // xml_tool_buffer
    run_id: String,
    events: Vec<Event>,
}

impl Ctx {
    #[inline]
    fn push(&mut self, ev_type: &'static str, content: String) {
        self.events.push(Event {
            ev_type,
            content,
            run_id: self.run_id.clone(),
        });
    }

    #[inline]
    fn emit_content(&mut self, s: String) {
        if !self.het && s.chars().any(|c| !c.is_whitespace()) {
            self.het = true;
        }
        self.push("content", s);
    }

    /// Consume `tag.len()` bytes from the front of the buffer.
    /// Safe because all tags are valid UTF-8 and always at char boundaries.
    #[inline]
    fn consume_tag(&mut self, tag: &str) {
        self.buf = self.buf[tag.len()..].to_string();
    }

    /// Pop and return the first character (by Unicode scalar) from the buffer.
    #[inline]
    fn pop_char(&mut self) -> String {
        let n = self.buf.chars().next().map(|c| c.len_utf8()).unwrap_or(0);
        if n == 0 {
            return String::new();
        }
        let ch = self.buf[..n].to_string();
        self.buf = self.buf[n..].to_string();
        ch
    }

    /// Return true if the current buffer could be the start (prefix) of any tag.
    /// Used to avoid emitting incomplete tag bytes as content.
    #[inline]
    fn could_be_partial(&self, tags: &[&str]) -> bool {
        let buf = self.buf.as_str();
        !buf.is_empty()
            && tags
                .iter()
                .any(|t| t.starts_with(buf) && buf.len() < t.len())
    }

    /// Flush the xml_tool_buffer as a raw tool call event for Python to resolve.
    fn flush_xml_tool(&mut self) {
        if !self.xtb.is_empty() {
            let xml = std::mem::take(&mut self.xtb);
            self.push("tool_call_raw_xml", xml);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the minimum byte index of `<`, `` ` ``, and optionally `{` in `s`.
#[inline]
fn find_first_special(s: &str, include_brace: bool) -> Option<usize> {
    let lt = s.find('<');
    let bt = s.find('`');
    let cu = if include_brace { s.find('{') } else { None };
    [lt, bt, cu].iter().filter_map(|x| *x).min()
}

// ── State handlers ────────────────────────────────────────────────────────────
// Each returns `true` to continue the outer loop, `false` to break.

fn handle_content(ctx: &mut Ctx) -> bool {
    let include_brace = !ctx.het;

    // Fast path: no special chars in buffer at all.
    if !ctx.buf.contains('<')
        && !ctx.buf.contains('`')
        && (!include_brace || !ctx.buf.contains('{'))
    {
        let s = std::mem::take(&mut ctx.buf);
        ctx.emit_content(s);
        return false;
    }

    match find_first_special(&ctx.buf, include_brace) {
        None => {
            let s = std::mem::take(&mut ctx.buf);
            ctx.emit_content(s);
            false
        }
        Some(0) => handle_content_dispatch(ctx),
        Some(cutoff) => {
            // Emit plain text up to the first special char.
            let text = ctx.buf[..cutoff].to_string();
            ctx.buf = ctx.buf[cutoff..].to_string();
            ctx.emit_content(text);
            true
        }
    }
}

fn handle_content_dispatch(ctx: &mut Ctx) -> bool {
    // Naked JSON: only when no text has been emitted yet.
    if !ctx.het && ctx.buf.starts_with('{') {
        ctx.buf = ctx.buf[1..].to_string();
        ctx.state = ST_NAKED_JSON;
        ctx.json_depth = 1;
        ctx.xtb = String::new();
        ctx.push("call_arguments", "{".to_string());
        return true;
    }

    // Try each tag in the ordered table.
    for (tag, new_state) in CONTENT_TAG_TABLE {
        if ctx.buf.starts_with(tag) {
            if let Some(ns) = new_state {
                ctx.state = *ns;
                if matches!(
                    ctx.state,
                    ST_FC | ST_TOOL_CALL_XML | ST_TOOL_CODE_XML | ST_MD_JSON_BLOCK
                ) {
                    ctx.xtb = String::new();
                }
            }
            ctx.consume_tag(tag);
            return true;
        }
    }

    // Check whether the buffer is a partial prefix of any known tag.
    // If so, wait for more data rather than emitting garbage.
    let tag_strs: Vec<&str> = CONTENT_TAG_TABLE.iter().map(|(t, _)| *t).collect();
    if ctx.could_be_partial(&tag_strs) {
        return false;
    }

    // Not a tag, not a partial — emit the first character and move on.
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.emit_content(ch);
    }
    true
}

fn handle_think_plan_decision(ctx: &mut Ctx) -> bool {
    let (end_tag, ev_type): (&str, &'static str) = match ctx.state {
        ST_THINK => (TH_END, "reasoning"),
        ST_PLAN => (PLAN_END, "plan"),
        _ => (DEC_END, "decision"),
    };

    // Fast path: no potential tag boundary.
    if !ctx.buf.contains('<') {
        let s = std::mem::take(&mut ctx.buf);
        ctx.push(ev_type, s);
        return false;
    }

    // Emit content up to the first `<`.
    let lt_idx = ctx.buf.find('<').unwrap_or(0);
    if lt_idx > 0 {
        let s = ctx.buf[..lt_idx].to_string();
        ctx.buf = ctx.buf[lt_idx..].to_string();
        ctx.push(ev_type, s);
    }

    if ctx.buf.starts_with(end_tag) {
        ctx.consume_tag(end_tag);
        ctx.state = ST_CONTENT;
        return true;
    }

    // Partial end-tag in buffer — wait.
    if end_tag.starts_with(ctx.buf.as_str()) && ctx.buf.len() < end_tag.len() {
        return false;
    }

    // Not a match, not partial — emit one character.
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push(ev_type, ch);
    }
    true
}

fn handle_xml_tool(ctx: &mut Ctx) -> bool {
    let end_tag: &str = match ctx.state {
        ST_FC => FC_END,
        ST_TOOL_CALL_XML => TC_END,
        ST_TOOL_CODE_XML => TCODE_END,
        _ => MD_END,
    };

    if ctx.buf.starts_with(end_tag) {
        ctx.consume_tag(end_tag);
        ctx.state = ST_CONTENT;
        ctx.flush_xml_tool();
        return true;
    }

    // Partial end-tag — wait.
    if end_tag.starts_with(ctx.buf.as_str()) && ctx.buf.len() < end_tag.len() {
        return false;
    }

    // Scan to the first character that COULD be the start of the end-tag,
    // then emit everything before it into the tool buffer.
    let first_end_char = end_tag.chars().next().unwrap();
    match ctx.buf.find(first_end_char) {
        None => {
            let s = std::mem::take(&mut ctx.buf);
            ctx.xtb.push_str(&s);
            ctx.push("call_arguments", s);
            false
        }
        Some(0) => {
            let ch = ctx.pop_char();
            ctx.xtb.push_str(&ch);
            ctx.push("call_arguments", ch);
            true
        }
        Some(i) => {
            let s = ctx.buf[..i].to_string();
            ctx.buf = ctx.buf[i..].to_string();
            ctx.xtb.push_str(&s);
            ctx.push("call_arguments", s);
            true
        }
    }
}

fn handle_naked_json(ctx: &mut Ctx) -> bool {
    if ctx.buf.is_empty() {
        return false;
    }

    let mut byte_offset = 0usize;
    let mut complete = false;

    for ch in ctx.buf.chars() {
        if ch == '{' {
            ctx.json_depth += 1;
        } else if ch == '}' {
            ctx.json_depth -= 1;
        }
        byte_offset += ch.len_utf8();
        if ctx.json_depth == 0 {
            complete = true;
            break;
        }
    }

    let chunk = ctx.buf[..byte_offset].to_string();
    ctx.buf = ctx.buf[byte_offset..].to_string();
    ctx.push("call_arguments", chunk);

    if complete {
        ctx.state = ST_CONTENT;
    }
    true
}

fn handle_kimi_router(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(KIMI_SEC_END) {
        ctx.consume_tag(KIMI_SEC_END);
        ctx.state = ST_CONTENT;
        return true;
    }
    if ctx.buf.starts_with(KIMI_ARG_START) {
        ctx.consume_tag(KIMI_ARG_START);
        ctx.state = ST_KIMI_ARGS;
        return true;
    }
    if ctx.buf.starts_with(KIMI_TC_START) {
        ctx.consume_tag(KIMI_TC_START);
        return true;
    }
    if ctx.buf.starts_with(KIMI_TC_END) {
        ctx.consume_tag(KIMI_TC_END);
        return true;
    }
    if ctx.could_be_partial(&[KIMI_SEC_END, KIMI_ARG_START, KIMI_TC_START, KIMI_TC_END]) {
        return false;
    }
    ctx.pop_char();
    true
}

fn handle_kimi_args(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(KIMI_TC_END) {
        ctx.consume_tag(KIMI_TC_END);
        ctx.state = ST_KIMI_ROUTER;
        return true;
    }
    if KIMI_TC_END.starts_with(ctx.buf.as_str()) && ctx.buf.len() < KIMI_TC_END.len() {
        return false;
    }
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push("call_arguments", ch);
    }
    true
}

fn handle_unicode_router(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(UNICODE_TC_END) {
        ctx.consume_tag(UNICODE_TC_END);
        ctx.state = ST_CONTENT;
        return true;
    }
    if ctx.buf.starts_with(UNICODE_CALL_BEGIN) {
        ctx.consume_tag(UNICODE_CALL_BEGIN);
        ctx.state = ST_UNICODE_TOOL_PARSING;
        return true;
    }
    if ctx.buf.starts_with(UNICODE_CALL_END) {
        ctx.consume_tag(UNICODE_CALL_END);
        return true;
    }
    if ctx.could_be_partial(&[UNICODE_TC_END, UNICODE_CALL_BEGIN, UNICODE_CALL_END]) {
        return false;
    }
    ctx.pop_char();
    true
}

fn handle_unicode_parsing(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(UNICODE_SEP) {
        ctx.consume_tag(UNICODE_SEP);
        ctx.state = ST_UNICODE_TOOL_ARGS;
        return true;
    }
    if ctx.buf.starts_with(UNICODE_CALL_END) {
        ctx.consume_tag(UNICODE_CALL_END);
        ctx.state = ST_UNICODE_TOOL_ROUTER;
        return true;
    }
    if ctx.could_be_partial(&[UNICODE_SEP, UNICODE_CALL_END]) {
        return false;
    }
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push("call_arguments", ch);
    }
    true
}

fn handle_unicode_args(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(UNICODE_CALL_END) {
        ctx.consume_tag(UNICODE_CALL_END);
        ctx.state = ST_UNICODE_TOOL_ROUTER;
        return true;
    }
    if UNICODE_CALL_END.starts_with(ctx.buf.as_str()) && ctx.buf.len() < UNICODE_CALL_END.len() {
        return false;
    }
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push("call_arguments", ch);
    }
    true
}

fn handle_channel_reasoning(ctx: &mut Ctx) -> bool {
    if ctx.buf.starts_with(CH_FINAL) {
        ctx.consume_tag(CH_FINAL);
        ctx.state = ST_CONTENT;
        return true;
    }
    if ctx.buf.starts_with(CH_COMMENTARY) {
        ctx.consume_tag(CH_COMMENTARY);
        ctx.state = ST_CHANNEL_TOOL_META;
        return true;
    }
    if ctx.could_be_partial(&[CH_FINAL, CH_COMMENTARY]) {
        return false;
    }
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push("reasoning", ch);
    }
    true
}

fn handle_channel_tool_meta(ctx: &mut Ctx) -> bool {
    if let Some(pos) = ctx.buf.find(MSG_TAG) {
        ctx.buf = ctx.buf[pos + MSG_TAG.len()..].to_string();
        ctx.state = ST_CHANNEL_TOOL_PAYLOAD;
        true
    } else if ctx.buf.contains(CH_FINAL) {
        ctx.state = ST_CONTENT;
        ctx.buf = String::new();
        true
    } else {
        false // wait for more data
    }
}

fn handle_channel_tool_payload(ctx: &mut Ctx) -> bool {
    for tag in [CALL_TAG, CH_FINAL, CH_ANALYSIS] {
        if ctx.buf.starts_with(tag) {
            ctx.state = if tag == CH_ANALYSIS {
                ST_CHANNEL_REASONING
            } else {
                ST_CONTENT
            };
            ctx.consume_tag(tag);
            return true;
        }
    }
    if ctx.could_be_partial(&[CALL_TAG, CH_FINAL, CH_ANALYSIS]) {
        return false;
    }
    let ch = ctx.pop_char();
    if !ch.is_empty() {
        ctx.push("call_arguments", ch);
    }
    true
}

// ── Main dispatch loop ────────────────────────────────────────────────────────

fn process(ctx: &mut Ctx) {
    loop {
        if ctx.buf.is_empty() {
            break;
        }

        let cont = match ctx.state {
            ST_CONTENT => handle_content(ctx),
            ST_THINK | ST_PLAN | ST_DECISION => handle_think_plan_decision(ctx),
            ST_FC | ST_TOOL_CALL_XML | ST_TOOL_CODE_XML | ST_MD_JSON_BLOCK => {
                handle_xml_tool(ctx)
            }
            ST_NAKED_JSON => handle_naked_json(ctx),
            ST_KIMI_ROUTER => handle_kimi_router(ctx),
            ST_KIMI_ARGS => handle_kimi_args(ctx),
            ST_UNICODE_TOOL_ROUTER => handle_unicode_router(ctx),
            ST_UNICODE_TOOL_PARSING => handle_unicode_parsing(ctx),
            ST_UNICODE_TOOL_ARGS => handle_unicode_args(ctx),
            ST_CHANNEL_REASONING => handle_channel_reasoning(ctx),
            ST_CHANNEL_TOOL_META => handle_channel_tool_meta(ctx),
            ST_CHANNEL_TOOL_PAYLOAD => handle_channel_tool_payload(ctx),
            _ => {
                // Unknown state: flush as content and stop.
                let s = std::mem::take(&mut ctx.buf);
                ctx.emit_content(s);
                false
            }
        };

        if !cont {
            break;
        }
    }
}

// ── PyO3 bindings ─────────────────────────────────────────────────────────────

/// Drop-in replacement for the C extension `process_buffer` function.
///
/// Signature is identical to `delta_normalizer_core.process_buffer` so the
/// Python wrapper (`delta_normalizer.py`) requires minimal changes.
#[pyfunction]
fn process_buffer(
    py: Python<'_>,
    buffer: String,
    state: u8,
    json_depth: i32,
    has_emitted_text: i32,
    xml_tool_buffer: String,
    run_id: String,
) -> PyResult<(PyObject, String, u8, i32, i32, String)> {
    let mut ctx = Ctx {
        buf: buffer,
        state,
        json_depth,
        het: has_emitted_text != 0,
        xtb: xml_tool_buffer,
        run_id,
        events: Vec::new(),
    };

    process(&mut ctx);

    let py_list = PyList::empty(py);
    for ev in &ctx.events {
        let d = PyDict::new(py);
        d.set_item("type", ev.ev_type)?;
        d.set_item("content", &ev.content)?;
        d.set_item("run_id", &ev.run_id)?;
        py_list.append(d)?;
    }

    Ok((
        py_list.into(),
        ctx.buf,
        ctx.state,
        ctx.json_depth,
        ctx.het as i32,
        ctx.xtb,
    ))
}

#[pymodule]
fn delta_normalizer_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_buffer, m)?)?;

    // Export state constants — same values as C extension for drop-in compat.
    m.add("ST_CONTENT", ST_CONTENT)?;
    m.add("ST_THINK", ST_THINK)?;
    m.add("ST_PLAN", ST_PLAN)?;
    m.add("ST_DECISION", ST_DECISION)?;
    m.add("ST_FC", ST_FC)?;
    m.add("ST_TOOL_CALL_XML", ST_TOOL_CALL_XML)?;
    m.add("ST_TOOL_CODE_XML", ST_TOOL_CODE_XML)?;
    m.add("ST_MD_JSON_BLOCK", ST_MD_JSON_BLOCK)?;
    m.add("ST_NAKED_JSON", ST_NAKED_JSON)?;
    m.add("ST_KIMI_ROUTER", ST_KIMI_ROUTER)?;
    m.add("ST_KIMI_ARGS", ST_KIMI_ARGS)?;
    m.add("ST_UNICODE_TOOL_ROUTER", ST_UNICODE_TOOL_ROUTER)?;
    m.add("ST_UNICODE_TOOL_PARSING", ST_UNICODE_TOOL_PARSING)?;
    m.add("ST_UNICODE_TOOL_ARGS", ST_UNICODE_TOOL_ARGS)?;
    m.add("ST_CHANNEL_REASONING", ST_CHANNEL_REASONING)?;
    m.add("ST_CHANNEL_TOOL_META", ST_CHANNEL_TOOL_META)?;
    m.add("ST_CHANNEL_TOOL_PAYLOAD", ST_CHANNEL_TOOL_PAYLOAD)?;

    Ok(())
}
