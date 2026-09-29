import asyncio
import inspect
from types import SimpleNamespace

from entities_api.platform_tools.tool_reigistry.research_worker import (
    RESEARCH_WORKER_ASSISTANT_TOOLS,
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


def test_research_worker_assistant_declaration_passes_namespace_guard():
    # This is the exact boundary that failed in integration.
    model = AssistantCreate(
        name="test-research-worker",
        model="test-model",
        tools=RESEARCH_WORKER_ASSISTANT_TOOLS,
        web_access=True,
    )

    assert model.tools == RESEARCH_WORKER_ASSISTANT_TOOLS

    assert {"type": "web_search"} in RESEARCH_WORKER_ASSISTANT_TOOLS

    declared_function_names = {
        tool["function"]["name"]
        for tool in RESEARCH_WORKER_ASSISTANT_TOOLS
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    assert {
        "read_scratchpad",
        "append_scratchpad",
    }.issubset(declared_function_names)

    # Expanded platform function names must NOT cross the persisted
    # AssistantCreate custom-function namespace.
    assert not {
        "perform_web_search",
        "read_web_page",
        "search_web_page",
        "scroll_web_page",
    }.intersection(declared_function_names)


def test_runtime_worker_registry_still_contains_expanded_web_tools():
    names = {
        tool["function"]["name"]
        for tool in WORKER_TOOLS
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    assert {
        "perform_web_search",
        "read_web_page",
        "search_web_page",
        "scroll_web_page",
        "read_scratchpad",
        "append_scratchpad",
    }.issubset(names)


def test_ephemeral_worker_uses_assistant_capability_declaration():
    source = inspect.getsource(AssistantManager.create_ephemeral_worker_assistant)

    assert "tools=RESEARCH_WORKER_ASSISTANT_TOOLS" in source
    assert "tools=WORKER_TOOLS" not in source
    assert "tools=JUNIOR_ENGINEER_TOOLS" not in source


def test_research_worker_has_multi_turn_budget():
    assert RESEARCH_WORKER_MAX_TURNS > 1
    assert RESEARCH_WORKER_MAX_TURNS == 8

    source = inspect.getsource(AssistantManager.create_ephemeral_worker_assistant)

    assert "max_turns=RESEARCH_WORKER_MAX_TURNS" in source
