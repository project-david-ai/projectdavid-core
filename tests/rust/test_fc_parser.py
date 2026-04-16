"""
fc_parser integration shim
--------------------------
Drop-in replacement for the parse_and_set_function_calls() hot path
in ToolRoutingMixin.

USAGE IN tool_routing_mixin.py
-------------------------------
Replace the body of parse_and_set_function_calls() with:

    from fc_parser import parse_function_calls as _rust_parse

    def parse_and_set_function_calls(
        self, accumulated_content: str, assistant_reply: str
    ) -> List[Dict]:
        from src.api.entities_api.orchestration.mixins.json_utils_mixin import JsonUtilsMixin
        if not isinstance(self, JsonUtilsMixin):
            raise TypeError("ToolRoutingMixin must be mixed with JsonUtilsMixin")

        results = _rust_parse(accumulated_content, assistant_reply)

        if results:
            LOG.info(f"L3-PARSER (Rust) ▸ Detected batch of {len(results)} tool(s).")
            self.set_tool_response_state(True)
            self.set_function_call_state(results)
            return results

        LOG.debug("L3-PARSER (Rust) ✗ nothing found")
        return []

BUILD
-----
From the fc_parser/ directory (requires Rust + maturin):

    pip install maturin
    maturin develop --release          # installs into current venv
    # or for a wheel:
    maturin build --release

The compiled .so/.pyd will be importable as `import fc_parser`.
"""

# Verify the extension is importable and functional
if __name__ == "__main__":
    try:
        from fc_parser import parse_function_calls
    except ImportError:
        print("fc_parser not built yet — run: maturin develop --release")
        raise

    # --- Test 1: Primary path — well-formed <fc> tag ---
    acc = '<fc>{"name": "web_search", "arguments": {"query": "vllm scaling"}}</fc>'
    result = parse_function_calls(acc, "")
    assert len(result) == 1
    assert result[0]["name"] == "web_search"
    assert result[0]["arguments"]["query"] == "vllm scaling"
    assert result[0]["id"].startswith("call_")
    print(f"✓ Test 1 passed: {result[0]}")

    # --- Test 2: Plan block stripped before scan ---
    acc = '<plan>Step 1: search</plan><fc>{"name": "perform_web_search", "arguments": {"query": "test"}}</fc>'
    result = parse_function_calls(acc, "")
    assert len(result) == 1
    assert result[0]["name"] == "perform_web_search"
    print(f"✓ Test 2 passed: plan stripped correctly")

    # --- Test 3: String-encoded arguments normalised ---
    acc = '<fc>{"name": "code_interpreter", "arguments": "{\\"code\\": \\"print(1)\\"}"}</fc>'
    result = parse_function_calls(acc, "")
    assert len(result) == 1
    assert isinstance(result[0]["arguments"], dict)
    assert result[0]["arguments"]["code"] == "print(1)"
    print(f"✓ Test 3 passed: string arguments normalised")

    # --- Test 4: Batch — multiple tools in one turn ---
    acc = (
        '<fc>{"name": "read_web_page", "arguments": {"url": "https://example.com"}}</fc>'
        '<fc>{"name": "append_scratchpad", "arguments": {"content": "found it"}}</fc>'
    )
    result = parse_function_calls(acc, "")
    assert len(result) == 2
    assert result[0]["name"] == "read_web_page"
    assert result[1]["name"] == "append_scratchpad"
    print(f"✓ Test 4 passed: batch of {len(result)} tools")

    # --- Test 5: Loose fallback — bare JSON object, no <fc> tags ---
    reply = '{"name": "file_search", "arguments": {"query": "project david"}}'
    result = parse_function_calls("", reply)
    assert len(result) == 1
    assert result[0]["name"] == "file_search"
    print(f"✓ Test 5 passed: loose fallback")

    # --- Test 6: No tool calls — returns empty list ---
    result = parse_function_calls(
        "This is a normal text reply.", "This is a normal text reply."
    )
    assert result == []
    print(f"✓ Test 6 passed: empty list on no tool calls")

    # --- Test 7: Smart quotes repaired ---
    acc = "<fc>{\u201cname\u201d: \u201cweb_search\u201d, \u201carguments\u201d: {\u201cquery\u201d: \u201ctest\u201d}}</fc>"
    result = parse_function_calls(acc, "")
    assert len(result) == 1
    assert result[0]["name"] == "web_search"
    print(f"✓ Test 7 passed: smart quotes repaired")

    print("\n✅ All tests passed.")
