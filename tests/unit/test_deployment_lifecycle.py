from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum

from src.api.training.services.deployment_service import DeploymentService


def _make_service() -> DeploymentService:
    service = DeploymentService.__new__(DeploymentService)
    service.db = MagicMock()
    service.registry = MagicMock()
    return service


def test_mark_deployments_cancelling_preserves_rows_and_changes_state():
    service = _make_service()

    active = SimpleNamespace(
        id="dep_active",
        status=StatusEnum.active,
        last_seen=0,
    )
    pending = SimpleNamespace(
        id="dep_pending",
        status=StatusEnum.pending,
        last_seen=0,
    )

    query = MagicMock()
    query.filter.return_value.all.return_value = [active, pending]

    deployment_ids = service._mark_deployments_cancelling(query)

    assert deployment_ids == ["dep_active", "dep_pending"]

    assert active.status == StatusEnum.cancelling
    assert pending.status == StatusEnum.cancelling

    assert active.last_seen > 0
    assert pending.last_seen > 0

    service.db.commit.assert_called_once()


@pytest.mark.parametrize(
    "status",
    [
        StatusEnum.pending,
        StatusEnum.active,
        StatusEnum.cancelling,
    ],
)
def test_activation_barrier_rejects_blocking_deployment_states(status):
    service = _make_service()

    service._get_blocking_deployments = MagicMock(
        return_value=[
            SimpleNamespace(
                id="dep_existing",
                status=status,
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        service._ensure_activation_allowed()

    assert exc_info.value.status_code == 409

    detail = exc_info.value.detail

    assert detail["deployments"] == [
        {
            "deployment_id": "dep_existing",
            "status": status.value,
        }
    ]


def test_activation_barrier_allows_activation_when_nothing_is_blocking():
    service = _make_service()

    service._get_blocking_deployments = MagicMock(return_value=[])

    service._ensure_activation_allowed()


def test_deactivate_base_model_transitions_deployment_to_cancelling():
    service = _make_service()

    base_model = SimpleNamespace(id="bm_test")

    service.registry.resolve.return_value = base_model
    service._mark_deployments_cancelling = MagicMock(return_value=["dep_test"])

    result = service.deactivate_base_model("Qwen/Test-Model")

    service.registry.resolve.assert_called_once_with("Qwen/Test-Model")
    service._mark_deployments_cancelling.assert_called_once()

    assert result == {
        "status": "cancelling",
        "base_model_id": "bm_test",
        "deployment_ids": ["dep_test"],
    }


def test_deactivate_base_model_is_already_cancelled_when_no_runtime_exists():
    service = _make_service()

    base_model = SimpleNamespace(id="bm_test")

    service.registry.resolve.return_value = base_model
    service._mark_deployments_cancelling = MagicMock(return_value=[])

    result = service.deactivate_base_model("Qwen/Test-Model")

    assert result == {
        "status": "cancelled",
        "base_model_id": "bm_test",
        "deployment_ids": [],
    }


def test_node_capacity_uses_actual_available_gpu_not_total_gpu():
    service = _make_service()

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": {
            "result": {
                "result": [
                    {
                        "node_id": "node_test",
                        "resources_total": {
                            "GPU": 1.0,
                        },
                        "resources_available": {
                            "GPU": 0.0,
                        },
                    }
                ]
            }
        }
    }

    with patch(
        "src.api.training.services.deployment_service.httpx.get",
        return_value=response,
    ):
        with pytest.raises(HTTPException) as exc_info:
            service._check_node_capacity(
                "node_test",
                tensor_parallel_size=1,
            )

    assert exc_info.value.status_code == 507
    assert "Available: 0" in exc_info.value.detail


@pytest.mark.parametrize(
    "node_payload",
    [
        {
            "node_id": "node_test",
            "resources_total": {
                "GPU": 1.0,
            },
        },
        {
            "node_id": "node_test",
            "resources_total": {
                "GPU": 1.0,
            },
            "resources_available": None,
        },
    ],
)
def test_node_capacity_fails_open_when_ray_omits_available_resources(
    node_payload,
):
    service = _make_service()

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": {
            "result": {
                "result": [
                    node_payload,
                ]
            }
        }
    }

    with patch(
        "src.api.training.services.deployment_service.httpx.get",
        return_value=response,
    ):
        service._check_node_capacity(
            "node_test",
            tensor_parallel_size=1,
        )
