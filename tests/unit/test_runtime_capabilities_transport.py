from __future__ import annotations

import importlib
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

deployments_router = importlib.import_module(
    "src.api.training.routers.deployments_router"
)
from src.api.training.services.deployment_service import DeploymentService


def test_service_returns_runtime_capability_payload():
    payload = {
        "schema_version": 1,
        "project_david_version": "1.47.1",
        "backend": {
            "id": "vllm",
            "version": "0.10.1",
        },
    }

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload

    service = DeploymentService.__new__(DeploymentService)

    with patch(
        "src.api.training.services." "deployment_service.httpx.get",
        return_value=response,
    ) as request:
        result = service.get_runtime_capabilities()

    assert result == payload

    request.assert_called_once_with(
        "http://inference_worker:8000/runtime-capabilities",
        timeout=5.0,
    )


def test_service_maps_unavailable_runtime_to_503():
    service = DeploymentService.__new__(DeploymentService)

    with patch(
        "src.api.training.services." "deployment_service.httpx.get",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(HTTPException) as exc:
            service.get_runtime_capabilities()

    assert exc.value.status_code == 503
    assert exc.value.detail == "Inference runtime capabilities unavailable."


def test_service_rejects_non_object_payload():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = ["not", "an", "object"]

    service = DeploymentService.__new__(DeploymentService)

    with patch(
        "src.api.training.services." "deployment_service.httpx.get",
        return_value=response,
    ):
        with pytest.raises(HTTPException) as exc:
            service.get_runtime_capabilities()

    assert exc.value.status_code == 502
    assert exc.value.detail == "Inference runtime capability response is invalid."


def test_router_exposes_admin_runtime_capabilities():
    expected = {
        "schema_version": 1,
        "project_david_version": "1.47.1",
    }

    service = Mock()
    service.get_runtime_capabilities.return_value = expected

    db = object()

    with (
        patch(
            "src.api.training.routers." "deployments_router._require_admin"
        ) as require_admin,
        patch(
            "src.api.training.routers." "deployments_router.DeploymentService",
            return_value=service,
        ),
    ):
        result = deployments_router.get_runtime_capabilities(
            db=db,
            current_user_id="user_test",
        )

    assert result == expected

    require_admin.assert_called_once_with(
        "user_test",
        db,
    )
    service.get_runtime_capabilities.assert_called_once_with()


def test_router_registers_exact_get_route():
    matches = [
        route
        for route in deployments_router.router.routes
        if getattr(route, "path", None) == "/runtime-capabilities"
    ]

    assert len(matches) == 1
    assert matches[0].methods == {"GET"}
