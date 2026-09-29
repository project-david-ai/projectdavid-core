import asyncio
from types import SimpleNamespace

from src.api.entities_api.orchestration.mixins.scratchpad_mixin import ScratchpadMixin


class _ScratchpadService:
    def __init__(self):
        self.resolve_calls = []
        self.read_calls = []
        self.update_calls = []
        self.append_calls = []

    async def resolve_scratchpad_for_thread(
        self,
        thread_id,
        *,
        user_id,
    ):
        self.resolve_calls.append((thread_id, user_id))

        return SimpleNamespace(
            id="scratchpad_resolved",
            owner_id=user_id,
            thread_id=thread_id,
        )

    async def get_formatted_view(
        self,
        scratchpad_id,
        *,
        user_id,
    ):
        self.read_calls.append((scratchpad_id, user_id))

        return "SHARED_STATE"

    async def set_content(
        self,
        scratchpad_id,
        content,
        *,
        user_id,
    ):
        self.update_calls.append(
            (
                scratchpad_id,
                content,
                user_id,
            )
        )

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
        self.append_calls.append(
            (
                scratchpad_id,
                content,
                user_id,
            )
        )

        return SimpleNamespace(
            scratchpad_id=scratchpad_id,
            content=content,
        )


class _RunService:
    @staticmethod
    def retrieve_run(run_id):
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
        super().__init__()
        self._native_exec = _NativeExec()


async def _drain(generator):
    return [event async for event in generator]


def test_direct_id_read_skips_thread_resolution():
    harness = _Harness()

    asyncio.run(
        _drain(
            harness.handle_read_scratchpad(
                thread_id="thread_worker",
                scratchpad_id="scratchpad_shared",
                scratch_pad_thread="thread_parent",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={},
                tool_call_id="call_read",
                decision=None,
            )
        )
    )

    service = harness._native_exec.scratchpad_svc

    assert service.resolve_calls == []

    assert service.read_calls == [
        (
            "scratchpad_shared",
            "user_owner",
        )
    ]

    assert harness._scratchpad_id == "scratchpad_shared"

    # Tool output remains on the worker conversation thread.
    assert harness._native_exec.outputs[0]["thread_id"] == "thread_worker"


def test_direct_id_update_and_append_share_resource():
    harness = _Harness()

    asyncio.run(
        _drain(
            harness.handle_update_scratchpad(
                thread_id="thread_worker",
                scratchpad_id="scratchpad_shared",
                scratch_pad_thread="thread_parent",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={
                    "content": "PLAN",
                },
                tool_call_id="call_update",
                decision=None,
            )
        )
    )

    asyncio.run(
        _drain(
            harness.handle_append_scratchpad(
                thread_id="thread_worker",
                scratchpad_id="scratchpad_shared",
                scratch_pad_thread="thread_parent",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={
                    "note": "finding",
                },
                tool_call_id="call_append",
                decision=None,
            )
        )
    )

    service = harness._native_exec.scratchpad_svc

    assert service.resolve_calls == []

    assert service.update_calls == [
        (
            "scratchpad_shared",
            "PLAN",
            "user_owner",
        )
    ]

    assert service.append_calls == [
        (
            "scratchpad_shared",
            "finding",
            "user_owner",
        )
    ]


def test_thread_locator_remains_legacy_fallback():
    harness = _Harness()

    asyncio.run(
        _drain(
            harness.handle_read_scratchpad(
                thread_id="thread_worker",
                scratch_pad_thread="thread_parent",
                run_id="run_test",
                assistant_id="asst_worker",
                arguments_dict={},
                tool_call_id="call_read",
                decision=None,
            )
        )
    )

    service = harness._native_exec.scratchpad_svc

    assert service.resolve_calls == [
        (
            "thread_parent",
            "user_owner",
        )
    ]

    assert service.read_calls == [
        (
            "scratchpad_resolved",
            "user_owner",
        )
    ]

    assert harness._scratchpad_id == "scratchpad_resolved"
