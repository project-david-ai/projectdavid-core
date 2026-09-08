"""Standalone regressions for capacity-aware local vLLM context admission."""

import ast
import json
import os
import posixpath
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from sqlalchemy import Column, Integer, String
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class BaseModel(Base):
    __tablename__ = "base_models"
    id = Column(String, primary_key=True)
    endpoint = Column(String)


class InferenceDeployment(Base):
    __tablename__ = "inference_deployments"
    id = Column(String, primary_key=True)
    base_model_id = Column(String)
    fine_tuned_model_id = Column(String)
    status = Column(String)
    internal_hostname = Column(String)
    node_id = Column(String)
    last_seen = Column(Integer)
    max_model_len = Column(Integer)


ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "src/api/entities_api/services/inference_resolver.py"
WORKER = (
    ROOT / "src/api/entities_api/orchestration/workers/base_workers/vllm_raw_worker.py"
)
RUNTIME_ROOT = "/opt/projectdavid/model-hub/models"
VARIANT = "qwen3-8b-awq-4bit"
ENDPOINT = f"{RUNTIME_ROOT}/qwen3-8b/{VARIANT}/{'a' * 24}"
URL = "http://inference_worker:8000/vllm_dep_Test"


def load_resolver():
    tree = ast.parse(RESOLVER.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "logging_utility"
                for target in node.targets
            )
        )
    ]
    namespace = dict(
        os=os,
        posixpath=posixpath,
        re=re,
        Optional=Optional,
        Session=Session,
        BaseModel=BaseModel,
        InferenceDeployment=InferenceDeployment,
        StatusEnum=SimpleNamespace(active="active"),
        logging_utility=Mock(),
    )
    exec(compile(tree, str(RESOLVER), "exec"), namespace)
    return namespace["InferenceResolver"]


def load_worker_helpers():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLLMDefaultBaseWorker"
    )
    wanted = {
        "_compact_json_schema",
        "_compact_tool_schema",
        "_compact_local_tool_context",
        "_estimate_local_prompt_tokens",
        "_local_completion_limit",
        "_admit_local_context",
    }
    methods = [
        node
        for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    minimal = ast.ClassDef(
        name="VLLMDefaultBaseWorker",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[]))
    namespace = dict(
        Any=Any,
        Dict=Dict,
        List=List,
        json=json,
        LOG=Mock(),
    )
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace["VLLMDefaultBaseWorker"]


class ContextAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"MODEL_HUB_RUNTIME_ROOT": RUNTIME_ROOT},
        )
        self.env.start()
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.resolver = load_resolver()
        self.worker = load_worker_helpers()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.env.stop()

    def add_route(self, *, hostname=URL, max_model_len=2048):
        self.db.add(BaseModel(id="bm_Test", endpoint=ENDPOINT))
        self.db.add(
            InferenceDeployment(
                id="dep_Test",
                base_model_id="bm_Test",
                fine_tuned_model_id=None,
                status="active",
                internal_hostname=hostname,
                node_id="node_test",
                last_seen=123,
                max_model_len=max_model_len,
            )
        )
        self.db.commit()

    def test_resolved_route_carries_runtime_capacity(self):
        self.add_route()
        route = self.resolver.resolve_vllm_route(self.db, VARIANT)
        self.assertEqual(route["url"], URL)
        self.assertEqual(route["deployment_id"], "dep_Test")
        self.assertEqual(route["max_model_len"], 2048)

    def test_legacy_route_capacity_is_preserved(self):
        self.add_route(hostname="172.20.0.2", max_model_len=4096)
        route = self.resolver.resolve_vllm_route(self.db, "bm_Test")
        self.assertEqual(route["url"], "http://172.20.0.2:8000")
        self.assertEqual(route["max_model_len"], 4096)

    def test_compact_tool_schema_preserves_argument_contract(self):
        tool = {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Runs Python safely. Long operational prose follows.",
                "parameters": {
                    "type": "object",
                    "description": "wrapper prose",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "very long argument prose",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
        compact = self.worker._compact_tool_schema(tool)
        self.assertEqual(compact["function"]["name"], "code_interpreter")
        self.assertEqual(compact["function"]["parameters"]["required"], ["code"])
        self.assertEqual(
            compact["function"]["parameters"]["properties"]["code"]["type"],
            "string",
        )
        self.assertNotIn("description", compact["function"]["parameters"])
        self.assertNotIn(
            "description",
            compact["function"]["parameters"]["properties"]["code"],
        )

    def test_2048_window_caps_completion_at_256(self):
        self.assertEqual(
            self.worker._local_completion_limit(
                2048,
                2048,
            ),
            256,
        )

        self.assertEqual(
            self.worker._local_completion_limit(
                128,
                2048,
            ),
            128,
        )

    def test_2048_window_elastically_shrinks_completion(self):
        messages = [
            {
                "role": "system",
                "content": "s" * 5200,
            },
            {
                "role": "user",
                "content": "Reply with exactly: pong",
            },
        ]

        requested = self.worker._local_completion_limit(
            2048,
            2048,
        )

        admitted, completion = self.worker._admit_local_context(
            messages,
            context_window=2048,
            completion_tokens=requested,
        )

        estimated = self.worker._estimate_local_prompt_tokens(admitted)

        self.assertGreaterEqual(
            completion,
            32,
        )

        self.assertLess(
            completion,
            requested,
        )

        self.assertLessEqual(
            estimated + completion + 64,
            2048,
        )

    def test_admission_keeps_system_and_latest_user_then_trims_history(self):
        messages = [
            {
                "role": "system",
                "content": "s" * 600,
            },
            {
                "role": "user",
                "content": "old " * 700,
            },
            {
                "role": "assistant",
                "content": "answer " * 700,
            },
            {
                "role": "user",
                "content": "Reply with exactly: pong",
            },
        ]

        admitted, completion = self.worker._admit_local_context(
            messages,
            context_window=2048,
            completion_tokens=256,
        )

        self.assertGreaterEqual(
            completion,
            32,
        )

        self.assertEqual(
            admitted[0]["role"],
            "system",
        )

        self.assertEqual(
            admitted[-1]["content"],
            "Reply with exactly: pong",
        )

        self.assertLess(
            len(admitted),
            len(messages),
        )

    def test_protected_overflow_fails_before_downstream_stream(self):
        messages = [
            {
                "role": "system",
                "content": "x" * 6000,
            },
            {
                "role": "user",
                "content": "pong",
            },
        ]

        with self.assertRaisesRegex(
            ValueError,
            "LOCAL_CONTEXT_CAPACITY_EXCEEDED",
        ):
            self.worker._admit_local_context(
                messages,
                context_window=2048,
                completion_tokens=256,
            )

        source = WORKER.read_text(encoding="utf-8")

        self.assertLess(
            source.index("LOCAL_CONTEXT_ADMISSION_V2"),
            source.index("self._stream_vllm_raw("),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
