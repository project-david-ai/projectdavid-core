
"""Isolated behavioral tests for request-scoped research delegation credentials.

The real DelegationMixin method is compiled from source, with external services
replaced by in-memory fakes. No Docker, network, database, or live secrets.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncGenerator, Optional


class _StatusEnum:
    completed = SimpleNamespace(value="completed")
    failed = SimpleNamespace(value="failed")


class _ScratchpadEvent:
    pass


class _WebStatusEvent:
    pass


def _delegation_type():
    source = (
        Path(__file__).resolve().parents[2]
        / "src/api/entities_api/orchestration/mixins/delegation_mixin.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))

    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DelegationMixin"
    )

    methods = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("_research_status", "handle_delegate_research_task")
    ]

    assert len(methods) == 2

    extracted = ast.Module(
        body=[
            ast.ClassDef(
                name="IsolatedDelegationMixin",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)

    namespace = {
        "json": json,
        "os": os,
        "LOG": logging.getLogger(__name__),
        "Optional": Optional,
        "AsyncGenerator": AsyncGenerator,
        "StatusEnum": _StatusEnum,
        "ScratchpadEvent": _ScratchpadEvent,
        "WebStatusEvent": _WebStatusEvent,
        "_TERMINAL_RUN_STATES": {
            "completed",
            "failed",
            "cancelled",
            "expired",
        },
    }

    exec(compile(extracted, str(source), "exec"), namespace)
    return namespace["IsolatedDelegationMixin"]


def _harness():
    instance = _delegation_type()()

    observations = SimpleNamespace(
        sdk_keys=[],
        provider_setups=[],
        child_runs=[],
        outputs=[],
        cleaned=[],
        created_workers=[],
        delegations=[],
        retired=[],
        action_statuses=[],
        parent_metadata={
            "delegated_model": "hyperbolic/deepseek-v4-flash-0731"
        },
    )

    async def create_action(**kwargs):
        return SimpleNamespace(id="action_1")

    async def retrieve_run(run_id):
        if run_id == "run_child":
            return SimpleNamespace(
                id=run_id,
                user_id="user_test",
                status="completed",
                last_error=None,
                meta_data={},
            )

        return SimpleNamespace(
            id=run_id,
            user_id="user_test",
            status="in_progress",
            meta_data=observations.parent_metadata,
        )

    async def update_action_status(action_id, status):
        observations.action_statuses.append((action_id, status))

    async def link_research_delegation(**kwargs):
        observations.delegations.append(kwargs)

    async def retire_research_worker_actor(**kwargs):
        observations.retired.append(kwargs)

    instance._native_exec = SimpleNamespace(
        create_action=create_action,
        retrieve_run=retrieve_run,
        update_action_status=update_action_status,
        link_research_delegation=link_research_delegation,
        retire_research_worker_actor=retire_research_worker_actor,
    )

    async def create_worker():
        observations.created_workers.append(True)
        return SimpleNamespace(
            id=f"worker_{len(observations.created_workers)}",
            name="research_worker",
        )

    async def create_thread():
        return SimpleNamespace(id="thread_child")

    async def create_message(thread_id, content, assistant_id):
        return SimpleNamespace(id="message_child")

    async def create_run(
        assistant_id,
        thread_id,
        meta_data,
        model,
<<<<<<< HEAD
        *,
        trusted_metadata=False,
    ):
        observations.child_runs.append(
            {
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "meta_data": meta_data.copy(),
                "model": model,
                "trusted_metadata": trusted_metadata,
            }
        )
=======
        trusted_metadata=None,
    ):
        observations.child_runs.append(((meta_data or {}).copy(), model))
>>>>>>> f0f28181 (fix(core): restore execution ledger migration and complete files integration)
        return SimpleNamespace(id="run_child")

    async def retrieve_run(run_id):
        if run_id == "run_child":
            return SimpleNamespace(
                id=run_id,
                user_id="user_test",
                status="completed",
                last_error=None,
            )

        return SimpleNamespace(
            id=run_id,
            user_id="user_test",
            meta_data=observations.parent_metadata,
        )

    async def cleanup(assistant_id, thread_id, delete_thread):
        observations.cleaned.append(assistant_id)

    async def submit_output(**kwargs):
        observations.outputs.append(kwargs)

    async def fake_events(*args, **kwargs):
        yield SimpleNamespace(
            content="Result from worker",
            reasoning=None,
        )

    class Stream:
        def setup(self, **kwargs):
            observations.provider_setups.append(kwargs)

        def stream_events(self, **kwargs):
            return iter(())

    def make_sdk(api_key, base_url):
        observations.sdk_keys.append(api_key)
        return SimpleNamespace(
            synchronous_inference_stream=Stream()
        )

    instance.create_ephemeral_worker_assistant = create_worker
    instance.create_ephemeral_thread = create_thread
    instance.create_ephemeral_message = create_message
    instance.create_ephemeral_run = create_run
    instance._ephemeral_clean_up = cleanup
    instance.submit_tool_output = submit_output
    instance._stream_sync_generator = fake_events
    instance._create_project_david_client = make_sdk

    instance._batfish_owner_user_id = "user_test"
    instance._delete_ephemeral_thread = False

    return instance, observations


async def _call(instance, **keys):
    return [
        json.loads(payload)
        async for payload in instance.handle_delegate_research_task(
            thread_id="thread_parent",
            run_id="run_parent",
            assistant_id="assistant_parent",
            arguments_dict={"task": "find sources"},
            tool_call_id="toolcall_1",
            decision=None,
            **keys,
        )
    ]


def test_two_research_delegations_reuse_request_credentials():
    async def scenario():
        instance, observed = _harness()

        credential_pairs = [
            ("provider_first", "project_first"),
            ("provider_second", "project_second"),
        ]

        for provider_key, project_key in credential_pairs:
            events = await _call(
                instance,
                inference_api_key=provider_key,
                project_david_api_key=project_key,
            )
            assert events[-1]["state"] == "completed"

        assert observed.sdk_keys == [
            "project_first",
            "project_second",
        ]

        assert [
            setup["api_key"]
            for setup in observed.provider_setups
        ] == [
            "provider_first",
            "provider_second",
        ]

        assert len(observed.outputs) == 2
        assert all(
            output["is_error"] is False
            for output in observed.outputs
        )

        assert observed.cleaned == [
            "worker_1",
            "worker_2",
        ]

        assert len(observed.child_runs) == 2
        assert len(observed.delegations) == 2
        assert len(observed.retired) == 2

        for index, child in enumerate(observed.child_runs):
            worker_id = f"worker_{index + 1}"

            assert child["assistant_id"] == worker_id
            assert child["thread_id"] == "thread_child"
            assert child["trusted_metadata"] is True

            assert child["model"] == (
                "hyperbolic/deepseek-v4-flash-0731"
            )

            assert child["meta_data"] == {
                "batfish_owner_user_id": "user_test",
                "scratch_pad_thread": "thread_parent",
                "root_run_id": "run_parent",
                "parent_run_id": "run_parent",
                "parent_action_id": "action_1",
            }

            assert "api_key" not in child["meta_data"]
            assert (
                "project_david_api_key"
                not in child["meta_data"]
            )

            assert observed.delegations[index] == {
                "parent_run_id": "run_parent",
                "child_run_id": "run_child",
                "parent_action_id": "action_1",
                "worker_assistant_id": worker_id,
                "worker_model": (
                    "hyperbolic/deepseek-v4-flash-0731"
                ),
            }

            assert observed.retired[index] == {
                "child_run_id": "run_child"
            }

        assert observed.action_statuses == [
            ("action_1", "processing"),
            ("action_1", "completed"),
            ("action_1", "processing"),
            ("action_1", "completed"),
        ]

        assert observed.parent_metadata == {
            "delegated_model": "hyperbolic/deepseek-v4-flash-0731"
        }

    asyncio.run(scenario())


def test_fails_closed_when_project_key_missing():
    async def scenario():
        instance, observed = _harness()

        events = await _call(
            instance,
            inference_api_key="provider_test_only",
        )

        assert any(
            event["state"] == "error"
            and "Project David API key" in event["activity"]
            for event in events
        )

        assert not observed.created_workers
        assert not observed.sdk_keys
        assert not observed.child_runs
        assert not observed.delegations

    asyncio.run(scenario())


def test_fails_closed_when_provider_key_missing():
    async def scenario():
        instance, observed = _harness()

        events = await _call(
            instance,
            project_david_api_key="project_test_only",
        )

        assert any(
            event["state"] == "error"
            and "Provider BYOK" in event["activity"]
            for event in events
        )

        assert not observed.created_workers
        assert not observed.provider_setups
        assert not observed.child_runs
        assert not observed.delegations

    asyncio.run(scenario())


def test_supervisor_persists_model_only():
    source_dir = (
        Path(__file__).resolve().parents[2]
        / "src/api/entities_api/orchestration/workers/base_workers"
    )

    for filename in ("qwen_base.py", "deepseek_base.py"):
        text = (source_dir / filename).read_text(
            encoding="utf-8"
        )

        assert (
            'meta_data={"delegated_model": pre_mapped_model}'
            in text
        )
        assert '"api_key": api_key,' not in text
        assert '"project_david_api_key": project_david_api_key,' not in text
        assert 'kwargs.get("project_david_api_key")' in text


def test_explicit_credential_wiring_is_present():
    root = Path(__file__).resolve().parents[2]

    core = (
        root
        / "src/api/entities_api/orchestration/engine/orchestrator_core.py"
    ).read_text(encoding="utf-8")

    router = (
        root
        / "src/api/entities_api/orchestration/mixins/tool_routing_mixin.py"
    ).read_text(encoding="utf-8")

    delegation = (
        root
        / "src/api/entities_api/orchestration/mixins/delegation_mixin.py"
    ).read_text(encoding="utf-8")

    assert (
        'project_david_api_key=kwargs.get("project_david_api_key")'
        in core
    )
    assert "project_david_api_key=project_david_api_key" in router
    assert "inference_api_key=api_key" in router

    assert 'run_obj.meta_data.get("api_key")' not in delegation
    assert (
        'run_obj.meta_data.get("project_david_api_key")'
        not in delegation
    )
    assert 'scrub_meta.pop("api_key"' not in delegation
