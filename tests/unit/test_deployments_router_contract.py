from unittest.mock import MagicMock, patch

from projectdavid_common.schemas.deployment_schemas import (
    DeactivateAllResponse,
    DeploymentDeactivationResponse,
)

from src.api.training.routers.deployments_router import (
    deactivate_all,
    deactivate_base_model,
    deactivate_fine_tuned_model,
)


def test_deactivate_all_uses_stateful_service_contract():
    db = MagicMock()
    service = MagicMock()

    service.deactivate_all.return_value = {
        "status": "cancelling",
        "deployment_ids": ["dep_test"],
        "message": "Teardown in progress.",
    }

    with (
        patch(
            "src.api.training.routers.deployments_router._require_admin"
        ) as require_admin,
        patch(
            "src.api.training.routers.deployments_router.DeploymentService",
            return_value=service,
        ),
    ):
        response = deactivate_all(
            db=db,
            current_user_id="user_admin",
        )

    require_admin.assert_called_once_with("user_admin", db)
    service.deactivate_all.assert_called_once_with()

    assert isinstance(response, DeactivateAllResponse)
    assert response.status == "cancelling"
    assert response.message == "Teardown in progress."


def test_deactivate_all_reports_cancelled_when_nothing_is_running():
    db = MagicMock()
    service = MagicMock()

    service.deactivate_all.return_value = {
        "status": "cancelled",
        "deployment_ids": [],
        "message": "No active local deployments require teardown.",
    }

    with (
        patch("src.api.training.routers.deployments_router._require_admin"),
        patch(
            "src.api.training.routers.deployments_router.DeploymentService",
            return_value=service,
        ),
    ):
        response = deactivate_all(
            db=db,
            current_user_id="user_admin",
        )

    assert isinstance(response, DeactivateAllResponse)
    assert response.status == "cancelled"
    assert response.message == ("No active local deployments require teardown.")


def test_deactivate_base_model_preserves_cancelling_status():
    db = MagicMock()
    service = MagicMock()

    service.deactivate_base_model.return_value = {
        "status": "cancelling",
        "base_model_id": "bm_test",
        "deployment_ids": ["dep_test"],
    }

    with (
        patch("src.api.training.routers.deployments_router._require_admin"),
        patch(
            "src.api.training.routers.deployments_router.DeploymentService",
            return_value=service,
        ),
    ):
        response = deactivate_base_model(
            model_ref="Qwen/Test-Model",
            db=db,
            current_user_id="user_admin",
        )

    service.deactivate_base_model.assert_called_once_with("Qwen/Test-Model")

    assert isinstance(response, DeploymentDeactivationResponse)
    assert response.status == "cancelling"
    assert response.base_model_id == "bm_test"


def test_deactivate_fine_tuned_model_preserves_cancelled_status():
    db = MagicMock()
    service = MagicMock()

    service.deactivate_model.return_value = {
        "status": "cancelled",
        "model_id": "ftm_test",
        "deployment_ids": [],
    }

    with (
        patch("src.api.training.routers.deployments_router._require_admin"),
        patch(
            "src.api.training.routers.deployments_router.DeploymentService",
            return_value=service,
        ),
    ):
        response = deactivate_fine_tuned_model(
            model_id="ftm_test",
            db=db,
            current_user_id="user_admin",
        )

    service.deactivate_model.assert_called_once_with("ftm_test")

    assert isinstance(response, DeploymentDeactivationResponse)
    assert response.status == "cancelled"
    assert response.model_id == "ftm_test"
