from src.api.entities_api.utilities.conversation_truncator import ConversationTruncator


def test_plain_same_role_messages_still_merge():
    conversation = [
        {"role": "user", "content": "alpha"},
        {"role": "user", "content": "beta"},
    ]

    result = ConversationTruncator.merge_consecutive_messages(conversation)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "alpha" in result[0]["content"]
    assert "beta" in result[0]["content"]


def test_parallel_tool_results_keep_distinct_call_ids():
    conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_scratchpad",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call_web",
                    "type": "function",
                    "function": {
                        "name": "perform_web_search",
                        "arguments": '{"query":"python docs"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": "scratchpad result",
            "tool_call_id": "call_read",
        },
        {
            "role": "tool",
            "content": "web result",
            "tool_call_id": "call_web",
        },
    ]

    result = ConversationTruncator.merge_consecutive_messages(conversation)

    assert len(result) == 3

    assert result[0]["tool_calls"][0]["id"] == "call_read"
    assert result[0]["tool_calls"][1]["id"] == "call_web"

    assert result[1] == {
        "role": "tool",
        "content": "scratchpad result",
        "tool_call_id": "call_read",
    }

    assert result[2] == {
        "role": "tool",
        "content": "web result",
        "tool_call_id": "call_web",
    }


def test_adjacent_assistant_tool_call_messages_are_atomic():
    conversation = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "tool_b",
                        "arguments": "{}",
                    },
                }
            ],
        },
    ]

    result = ConversationTruncator.merge_consecutive_messages(conversation)

    assert len(result) == 2
    assert result[0]["tool_calls"][0]["id"] == "call_a"
    assert result[1]["tool_calls"][0]["id"] == "call_b"


def test_explicit_tool_call_id_is_atomic_even_without_tool_role():
    conversation = [
        {
            "role": "platform",
            "content": "one",
            "tool_call_id": "call_one",
        },
        {
            "role": "platform",
            "content": "two",
            "tool_call_id": "call_two",
        },
    ]

    result = ConversationTruncator.merge_consecutive_messages(conversation)

    assert len(result) == 2
    assert result[0]["tool_call_id"] == "call_one"
    assert result[1]["tool_call_id"] == "call_two"
