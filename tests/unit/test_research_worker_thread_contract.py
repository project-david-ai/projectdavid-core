import asyncio
from types import SimpleNamespace

from entities_api.orchestration.mixins.context_mixin import ContextMixin
from entities_api.platform_tools.tool_reigistry.research_worker import (
    RESEARCH_WORKER_MAX_TURNS,
    WORKER_TOOLS,
)
from projectdavid_common.schemas.assistants_schema import AssistantCreate

from src.api.entities_api.orchestration.mixins.scratchpad_mixin import ScratchpadMixin
from src.api.entities_api.utilities.assistant_manager import AssistantManager


class _ScratchpadService:
    def __init__(self):
        self.resolve_thread = None
        self.resolve_user = None

        self.read_scratchpad = None
        self.read_user = None

        self.append_scratchpad = None
        self.append_user = None

        self.update_scratchpad = None
        self.update_user = None

    async def resolve_scratchpad_for_thread(
        self,
        thread_id,
        *,
        user_id,
    ):
        self.resolve_thread = thread_id
        self.resolve_user = user_id

        return SimpleNamespace(
            id="scratchpad_shared",
            owner_id=user_id,
            thread_id=thread_id,
        )

    async def get_formatted_view(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.read_scratchpad = scratchpad_id
        self.read_user = user_id

        return "SHARED_STATE"

    async def set_content(
        self,
        scratchpad_id,
        content,
        *,
        user_id,
    ):
        self.update_scratchpad = scratchpad_id
        self.update_user = user_id

        return SimpleNamespace(
            scratchpad_id=scratchpad_id,
            content=content,
        )

    async def append_entry(
        self,
        scratchpad_id,
        content,
        *,
        user_id,
    ):
        self.append_scratchpad = scratchpad_id
        self.append_user = user_id

        return SimpleNamespace(
            scratchpad_id=scratchpad_id,
            content=content,
        )


class _RunService:
    def retrieve_run(self, run_id):
        return SimpleNamespace(
            id=run_id,
            user_id="user_owner",
        )


class _NativeExec:
    def __init__(self):
        self.scratchpad_svc = _ScratchpadService()
        self.run_svc = _RunService()
        self.outputs = []

    async def create_action(self, **kwargs):
        return SimpleNamespace(id="act_test")

    async def update_action_status(self, **kwargs):
        return None

    async def submit_tool_output(self, **kwargs):
        self.outputs.append(kwargs)


class _Harness(ScratchpadMixin):
    def __init__(self):
        self._native_exec = _NativeExec()
        self._scratch_pad_thread = None


def test_shared_read_uses_shared_data_but_worker_tool_result():
    harness = _Harness()

    async def exercise():
        return [
            event
            async for event in harness.handle_read_scratchpad(
                thread_id="thread_worker",
                scratch_pad_thread="thread_shared",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={},
                tool_call_id="call_read",
                decision=None,
            )
        ]

    asyncio.run(exercise())

    assert harness._native_exec.scratchpad_svc.resolve_thread == "thread_shared"
    assert harness._native_exec.scratchpad_svc.read_user == "user_owner"
    assert harness._native_exec.scratchpad_svc.read_scratchpad == "scratchpad_shared"

    output = harness._native_exec.outputs[0]
    assert output["thread_id"] == "thread_worker"
    assert output["tool_call_id"] == "call_read"


def test_shared_append_uses_shared_data_but_worker_tool_result():
    harness = _Harness()

    async def exercise():
        return [
            event
            async for event in harness.handle_append_scratchpad(
                thread_id="thread_worker",
                scratch_pad_thread="thread_shared",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={"note": "worker note"},
                tool_call_id="call_append",
                decision=None,
            )
        ]

    asyncio.run(exercise())

    assert harness._native_exec.scratchpad_svc.append_scratchpad == "scratchpad_shared"
    assert harness._native_exec.scratchpad_svc.append_user == "user_owner"

    output = harness._native_exec.outputs[0]
    assert output["thread_id"] == "thread_worker"
    assert output["tool_call_id"] == "call_append"


def test_ephemeral_worker_uses_platform_capability_placeholders():
    class _CaptureNativeExec:
        def __init__(self):
            self.kwargs = None

        async def create_assistant(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(**kwargs)

    manager = AssistantManager()
    capture = _CaptureNativeExec()

    manager._native_exec_svc = capture

    asyncio.run(manager.create_ephemeral_worker_assistant(user_id="user_test"))

    assert capture.kwargs["tools"] == [
        {"type": "web_search"},
        {"type": "scratchpad"},
    ]

    assert capture.kwargs["web_access"] is True

    assert capture.kwargs["max_turns"] == RESEARCH_WORKER_MAX_TURNS


def test_research_worker_platform_capabilities_hydrate_at_context_boundary():
    resolved = ContextMixin._resolve_and_prioritize_platform_tools(
        [
            {"type": "web_search"},
            {"type": "scratchpad"},
        ]
    )

    names = {
        tool["function"]["name"]
        for tool in resolved
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    assert {
        "perform_web_search",
        "read_web_page",
        "search_web_page",
        "scroll_web_page",
        "read_scratchpad",
        "update_scratchpad",
        "append_scratchpad",
    }.issubset(names)


def test_scratchpad_platform_schema_hides_resource_identity():
    resolved = ContextMixin._resolve_and_prioritize_platform_tools(
        [{"type": "scratchpad"}]
    )

    expected_names = {
        "read_scratchpad",
        "update_scratchpad",
        "append_scratchpad",
    }

    actual_names = {tool["function"]["name"] for tool in resolved}

    assert actual_names == expected_names

    for tool in resolved:
        function = tool["function"]
        schema = function["parameters"]

        assert schema["additionalProperties"] is False

        properties = schema["properties"]

        assert "scratchpad_id" not in properties
        assert "thread_id" not in properties
        assert "user_id" not in properties
        assert "owner_id" not in properties


def test_research_worker_has_multi_turn_budget():
    assert RESEARCH_WORKER_MAX_TURNS > 1
    assert RESEARCH_WORKER_MAX_TURNS == 8
