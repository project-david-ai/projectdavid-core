use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Compiled regex — initialised once at module load, never re-compiled.
// ---------------------------------------------------------------------------

static FC_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<fc>\s*(?P<payload>\{.*?\})\s*</fc>").unwrap()
});

static PLAN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<plan>.*?</plan>").unwrap()
});

static FENCE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)```(?:json)?(.*?)```").unwrap()
});

static TRAILING_COMMA_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r",(\s*[}\]])").unwrap()
});

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: Value,
}

// ---------------------------------------------------------------------------
// Argument normalisation
// ---------------------------------------------------------------------------
fn normalize_arguments(payload: &mut Value) {
    if let Some(args) = payload.get("arguments") {
        if let Some(s) = args.as_str() {
            let cleaned = s.trim().trim_matches('`').trim();
            let cleaned = if cleaned.starts_with("json") {
                cleaned[4..].trim()
            } else {
                cleaned
            };
            if let Ok(parsed) = serde_json::from_str::<Value>(cleaned) {
                payload["arguments"] = parsed;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Heuristic JSON repair
// ---------------------------------------------------------------------------
fn try_parse_json(raw: &str) -> Option<Value> {
    let text = raw.trim();

    if let Ok(v) = serde_json::from_str::<Value>(text) {
        if v.is_object() {
            return Some(v);
        }
        if let Some(inner) = v.as_str() {
            if let Ok(inner_v) = serde_json::from_str::<Value>(inner) {
                if inner_v.is_object() {
                    return Some(inner_v);
                }
            }
        }
        return None;
    }

    let fixed = text
        .replace('\u{2018}', "'")
        .replace('\u{2019}', "'")
        .replace('\u{201C}', "\"")
        .replace('\u{201D}', "\"");
    let fixed = TRAILING_COMMA_REGEX.replace_all(&fixed, "$1");

    if let Ok(v) = serde_json::from_str::<Value>(&fixed) {
        if v.is_object() {
            return Some(v);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Validate a parsed object is a tool call
// ---------------------------------------------------------------------------
fn is_valid_tool_call(v: &Value) -> bool {
    let obj = match v.as_object() {
        Some(o) => o,
        None => return false,
    };
    match obj.get("name").and_then(|n| n.as_str()) {
        Some(s) if !s.trim().is_empty() => {}
        _ => return false,
    }
    match obj.get("arguments") {
        Some(Value::Object(_)) => true,
        Some(Value::String(_)) => true,
        _ => false,
    }
}

fn make_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().to_string().replace('-', "")[..8])
}

// ---------------------------------------------------------------------------
// Primary path: scan <fc>...</fc> tags
// ---------------------------------------------------------------------------
fn parse_fc_tags(accumulated_content: &str) -> Vec<ToolCall> {
    let body = PLAN_REGEX.replace_all(accumulated_content, "");
    let mut results = Vec::new();

    for cap in FC_REGEX.captures_iter(&body) {
        let raw_payload = &cap["payload"];
        let mut parsed = match try_parse_json(raw_payload) {
            Some(v) => v,
            None => continue,
        };
        if !is_valid_tool_call(&parsed) {
            continue;
        }
        normalize_arguments(&mut parsed);

        let id = parsed
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(make_call_id);

        let name = parsed["name"].as_str().unwrap_or("").to_string();
        let arguments = parsed
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        results.push(ToolCall { id, name, arguments });
    }
    results
}

// ---------------------------------------------------------------------------
// Loose fallback
// ---------------------------------------------------------------------------
fn parse_loose(assistant_reply: &str) -> Vec<ToolCall> {
    let text = FENCE_REGEX.replace_all(assistant_reply, "$1");
    let text = text
        .trim()
        .replace('\u{2018}', "'")
        .replace('\u{2019}', "'")
        .replace('\u{201C}', "\"")
        .replace('\u{201D}', "\"");
    let text = text.trim().to_string();

    if !(text.starts_with('{') && text.ends_with('}')) {
        return Vec::new();
    }

    let mut parsed = match try_parse_json(&text) {
        Some(v) => v,
        None => return Vec::new(),
    };
    if !is_valid_tool_call(&parsed) {
        return Vec::new();
    }
    normalize_arguments(&mut parsed);

    let id = parsed
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(make_call_id);

    let name = parsed["name"].as_str().unwrap_or("").to_string();
    let arguments = parsed
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    vec![ToolCall { id, name, arguments }]
}

// ---------------------------------------------------------------------------
// Convert serde_json::Value → Python object (pyo3 0.28 API)
// ---------------------------------------------------------------------------
fn value_to_pyobject<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    match v {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(b) => Ok((*b).into_pyobject(py)?.to_owned().into_any()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any())
            } else {
                Ok(n.to_string().into_pyobject(py)?.into_any())
            }
        }
        Value::String(s) => Ok(s.as_str().into_pyobject(py)?.into_any()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(value_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, value_to_pyobject(py, val)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

fn tool_call_to_pydict<'py>(py: Python<'py>, tc: &ToolCall) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &tc.id)?;
    dict.set_item("name", &tc.name)?;
    dict.set_item("arguments", value_to_pyobject(py, &tc.arguments)?)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Public Python-facing function (pyo3 0.28 module API)
// ---------------------------------------------------------------------------
#[pyfunction]
fn parse_function_calls<'py>(
    py: Python<'py>,
    accumulated_content: &str,
    assistant_reply: &str,
) -> PyResult<Bound<'py, PyList>> {
    let mut calls = parse_fc_tags(accumulated_content);
    if calls.is_empty() {
        calls = parse_loose(assistant_reply);
    }
    let list = PyList::empty(py);
    for tc in &calls {
        list.append(tool_call_to_pydict(py, tc)?)?;
    }
    Ok(list)
}

#[pymodule]
fn fc_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_function_calls, m)?)?;
    Ok(())
}
