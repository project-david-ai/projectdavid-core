"""Standalone resolver regressions: real SQLAlchemy queries, isolated ledger schema.

Run: python tests/test_model_hub_variant_routing.py
No API, Redis, GPU or production database is imported or contacted.
"""

import ast
import os
import posixpath
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
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
    # Substitute only external imports/logger; execute production resolver code.
    tree.body = [
        n
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "logging_utility" for t in n.targets
            )
        )
    ]
    ns = dict(
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
    exec(compile(tree, str(RESOLVER), "exec"), ns)
    return ns["InferenceResolver"]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MODEL_HUB_RUNTIME_ROOT": RUNTIME_ROOT})
        self.env.start()
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.resolver = load_resolver()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.env.stop()

    def add(self, endpoint=ENDPOINT, key="Test", status="active", url=URL, ftm=None):
        self.db.add(BaseModel(id="bm_" + key, endpoint=endpoint))
        self.db.add(
            InferenceDeployment(
                id="dep_" + key,
                base_model_id="bm_" + key,
                fine_tuned_model_id=ftm,
                status=status,
                internal_hostname=url,
                node_id="node_test",
                last_seen=123,
            )
        )
        self.db.commit()

    def resolve(self, tag=VARIANT):
        return self.resolver.resolve_vllm_url(self.db, tag)

    def test_registered_variant_with_and_without_prefix(self):
        self.add()
        for tag in (VARIANT, "vllm/" + VARIANT):
            self.assertEqual(self.resolve(tag), URL)

    def test_legacy_id_and_endpoint_routes(self):
        self.add(endpoint="Qwen/Qwen2.5-1.5B-Instruct-AWQ", ftm="ftm_Adapter")
        for tag in (
            "vllm/bm_Test",
            "vllm/dep_Test",
            "vllm/vllm_dep_Test",
            "vllm/ftm_Adapter",
            "vllm/Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        ):
            self.assertEqual(self.resolve(tag), URL)

    def test_bare_hostname_legacy_route(self):
        self.add(url="172.20.0.2")
        self.assertEqual(self.resolve("bm_Test"), "http://172.20.0.2:8000")

    def test_inactive_does_not_match(self):
        self.add(status="pending")
        self.assertIsNone(self.resolve())

    def test_inactive_revision_does_not_make_active_ambiguous(self):
        self.add()
        self.add(key="Other", endpoint=ENDPOINT[:-24] + "b" * 24, status="pending")
        self.assertEqual(self.resolve(), URL)

    def test_two_active_revisions_fail_closed(self):
        self.add()
        self.add(key="Other", endpoint=ENDPOINT[:-24] + "b" * 24)
        with self.assertRaisesRegex(ValueError, "MODEL_HUB_ROUTE_AMBIGUOUS"):
            self.resolve()

    def test_same_variant_across_models_fails_closed(self):
        self.add()
        self.add(key="Other", endpoint=ENDPOINT.replace("/qwen3-8b/", "/other/"))
        with self.assertRaisesRegex(ValueError, "MODEL_HUB_ROUTE_AMBIGUOUS"):
            self.resolve()

    def test_multiple_replicas_fail_closed(self):
        self.add()
        self.db.add(
            InferenceDeployment(
                id="dep_Second",
                base_model_id="bm_Test",
                status="active",
                internal_hostname=URL,
                last_seen=124,
            )
        )
        self.db.commit()
        with self.assertRaisesRegex(ValueError, "MODEL_HUB_ROUTE_AMBIGUOUS"):
            self.resolve()

    def test_adapter_is_not_implicit_base(self):
        self.add(ftm="ftm_Adapter")
        self.assertIsNone(self.resolve())
        self.assertEqual(self.resolve("ftm_Adapter"), URL)

    def test_missing_hostname_does_not_match(self):
        self.add(url=None)
        self.assertIsNone(self.resolve())

    def test_path_contract_rejects_lookalikes(self):
        endpoints = [
            ENDPOINT + "/extra",
            ENDPOINT + "/",
            ENDPOINT + " ",
            ENDPOINT.replace("/qwen3-8b/", "/../"),
            ENDPOINT.replace("/qwen3-8b/", "//"),
            ENDPOINT.replace("/models/", "/models-evil/"),
            ENDPOINT.replace("/opt/", "/tmp/"),
            ENDPOINT[:-24] + "revision-name",
            ENDPOINT.replace("/", "\\"),
            ENDPOINT.replace(VARIANT, VARIANT + "-other"),
        ]
        for i, endpoint in enumerate(endpoints):
            self.add(endpoint=endpoint, key=str(i))
        self.assertIsNone(self.resolve())

    def test_custom_root(self):
        with patch.dict(os.environ, {"MODEL_HUB_RUNTIME_ROOT": "/custom/root"}):
            self.add(endpoint=ENDPOINT.replace(RUNTIME_ROOT, "/custom/root"))
            self.assertEqual(self.resolve(), URL)

    def test_invalid_root_rejected(self):
        with patch.dict(os.environ, {"MODEL_HUB_RUNTIME_ROOT": "/"}):
            with self.assertRaises(ValueError):
                self.resolve()

    def test_unknown_variant_returns_none(self):
        self.add()
        self.assertIsNone(self.resolve("different-model"))

    def test_exact_endpoint_still_supported(self):
        self.add()
        self.assertEqual(self.resolve("vllm/" + ENDPOINT), URL)

    def test_prefix_only_removed_once(self):
        self.add(endpoint="org/vllm/model")
        self.assertEqual(self.resolve("vllm/org/vllm/model"), URL)

    def test_worker_fallback_guard(self):
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        guard = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "MODEL_ROUTE_UNAVAILABLE" in ast.unparse(n)
            and isinstance(n.test, ast.BoolOp)
        )
        for model, custom, mesh, fails in (
            (VARIANT, None, None, True),
            ("vllm/" + VARIANT, None, None, True),
            ("bm_Test", None, None, True),
            (VARIANT, None, URL, False),
            (VARIANT, URL, None, False),
            ("org/hf-model", None, None, False),
        ):
            ns = dict(
                model=model,
                custom_vllm_url=custom,
                mesh_resolved_url=mesh,
                InferenceResolver=self.resolver,
            )
            code = compile(ast.Module(body=[guard], type_ignores=[]), "<guard>", "exec")
            if fails:
                with self.assertRaisesRegex(ValueError, "MODEL_ROUTE_UNAVAILABLE"):
                    exec(code, ns)
            else:
                exec(code, ns)

    def test_worker_propagates_variant_resolution_errors(self):
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        handler = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler)
            and "Mesh Resolution Error" in ast.unparse(n)
        )
        for model, expected in ((VARIANT, True), ("org/hf-model", False)):
            code = ast.parse(
                "try:\n    raise ValueError('test')\nexcept Exception:\n    pass"
            )
            code.body[0].handlers = [handler]
            ns = dict(model=model, InferenceResolver=self.resolver, LOG=Mock())
            if expected:
                with self.assertRaisesRegex(ValueError, "test"):
                    exec(
                        compile(ast.fix_missing_locations(code), "<handler>", "exec"),
                        ns,
                    )
            else:
                exec(compile(ast.fix_missing_locations(code), "<handler>", "exec"), ns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
