from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from projectdavid_common.schemas.enums import StatusEnum


def _load_inference_worker(monkeypatch):
    fake_serve = SimpleNamespace(
        deployment=lambda **_kwargs: lambda cls: cls,
    )

    fake_ray = types.ModuleType("ray")
    fake_ray.serve = fake_serve
    fake_ray.cluster_resources = lambda: {"GPU": 1.0}
    fake_ray.available_resources = lambda: {"GPU": 1.0}

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    sys.modules.pop("src.api.training.inference_worker", None)

    module = importlib.import_module("src.api.training.inference_worker")

    return module


def _make_reconciler(module):
    reconciler = module.InferenceReconciler.__new__(module.InferenceReconciler)
    reconciler._active = {}
    return reconciler


def test_cancelling_deployment_requests_ray_deletion(monkeypatch):
    module = _load_inference_worker(monkeypatch)
    reconciler = _make_reconciler(module)

    db = MagicMock()

    deployment = SimpleNamespace(
        id="dep_old",
        status=StatusEnum.cancelling,
        internal_hostname="http://old",
        last_seen=0,
        tensor_parallel_size=1,
    )

    deployment_name = module._deployment_name(deployment.id)

    reconciler._get_db = MagicMock(return_value=db)
    reconciler._get_cancelling_deployments = MagicMock(
        side_effect=[
            [deployment],
            [deployment],
        ]
    )

    reconciler._get_serve_applications = MagicMock(
        return_value={deployment_name: SimpleNamespace(status="RUNNING")}
    )

    reconciler._delete_deployment = MagicMock(return_value=True)
    reconciler._deploy = MagicMock()

    reconciler.reconcile()

    reconciler._delete_deployment.assert_called_once_with(deployment_name)

    assert deployment.status == StatusEnum.cancelling
    reconciler._deploy.assert_not_called()
    db.close.assert_called_once()


def test_cancellation_completes_only_after_ray_and_gpu_are_free(
    monkeypatch,
):
    module = _load_inference_worker(monkeypatch)
    reconciler = _make_reconciler(module)

    db = MagicMock()

    deployment = SimpleNamespace(
        id="dep_old",
        status=StatusEnum.cancelling,
        internal_hostname="http://old",
        last_seen=0,
        tensor_parallel_size=1,
    )

    reconciler._active[deployment.id] = module._deployment_name(deployment.id)

    reconciler._get_db = MagicMock(return_value=db)

    reconciler._get_cancelling_deployments = MagicMock(
        side_effect=[
            [deployment],
            [],
        ]
    )

    reconciler._get_pending_deployments = MagicMock(return_value=[])

    reconciler._get_serve_applications = MagicMock(
        side_effect=[
            {},
            {},
            {},
        ]
    )

    reconciler._gpu_capacity_available = MagicMock(return_value=True)

    reconciler.reconcile()

    assert deployment.status == StatusEnum.cancelled
    assert deployment.internal_hostname is None
    assert deployment.last_seen > 0

    assert deployment.id not in reconciler._active

    reconciler._gpu_capacity_available.assert_called_once_with(deployment)

    db.commit.assert_called()
    db.close.assert_called_once()


def test_cancellation_waits_when_gpu_is_still_occupied(
    monkeypatch,
):
    module = _load_inference_worker(monkeypatch)
    reconciler = _make_reconciler(module)

    db = MagicMock()

    deployment = SimpleNamespace(
        id="dep_old",
        status=StatusEnum.cancelling,
        internal_hostname="http://old",
        last_seen=0,
        tensor_parallel_size=1,
    )

    reconciler._get_db = MagicMock(return_value=db)

    reconciler._get_cancelling_deployments = MagicMock(
        side_effect=[
            [deployment],
            [deployment],
        ]
    )

    reconciler._get_serve_applications = MagicMock(return_value={})

    reconciler._gpu_capacity_available = MagicMock(return_value=False)

    reconciler._deploy = MagicMock()

    reconciler.reconcile()

    assert deployment.status == StatusEnum.cancelling
    assert deployment.internal_hostname == "http://old"

    reconciler._deploy.assert_not_called()
    db.close.assert_called_once()


def test_orphan_teardown_blocks_new_deployment_same_poll(
    monkeypatch,
):
    module = _load_inference_worker(monkeypatch)
    reconciler = _make_reconciler(module)

    db = MagicMock()

    pending = SimpleNamespace(
        id="dep_new",
        status=StatusEnum.pending,
        internal_hostname=None,
        last_seen=0,
        tensor_parallel_size=1,
    )

    reconciler._get_db = MagicMock(return_value=db)

    reconciler._get_cancelling_deployments = MagicMock(
        side_effect=[
            [],
            [],
        ]
    )

    reconciler._get_pending_deployments = MagicMock(return_value=[pending])

    reconciler._get_serve_applications = MagicMock(
        return_value={"vllm_dep_orphan": SimpleNamespace(status="RUNNING")}
    )

    reconciler._delete_deployment = MagicMock(return_value=True)
    reconciler._deploy = MagicMock()

    reconciler.reconcile()

    reconciler._delete_deployment.assert_called_once_with("vllm_dep_orphan")

    reconciler._deploy.assert_not_called()
    db.close.assert_called_once()


def test_pending_becomes_active_only_when_ray_reports_running(
    monkeypatch,
):
    module = _load_inference_worker(monkeypatch)
    reconciler = _make_reconciler(module)

    db = MagicMock()

    deployment = SimpleNamespace(
        id="dep_new",
        status=StatusEnum.pending,
        internal_hostname="http://new",
        last_seen=0,
        tensor_parallel_size=1,
    )

    deployment_name = module._deployment_name(deployment.id)

    reconciler._get_db = MagicMock(return_value=db)

    reconciler._get_cancelling_deployments = MagicMock(
        side_effect=[
            [],
            [],
        ]
    )

    reconciler._get_pending_deployments = MagicMock(return_value=[deployment])

    running_application = SimpleNamespace(status="RUNNING")

    reconciler._get_serve_applications = MagicMock(
        return_value={deployment_name: running_application}
    )

    reconciler._deploy = MagicMock()

    reconciler.reconcile()

    assert deployment.status == StatusEnum.active
    assert deployment.last_seen > 0

    assert reconciler._active[deployment.id] == deployment_name

    reconciler._deploy.assert_not_called()
    db.commit.assert_called()
    db.close.assert_called_once()
