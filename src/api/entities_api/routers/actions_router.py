from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.actions_service import ActionService

router = APIRouter()
logging_utility = LoggingUtility()


@router.post("/actions", response_model=ValidationInterface.ActionRead)
def create_action(
    action: ValidationInterface.ActionCreate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to create a new action."
    )
    action_service = ActionService()
    try:
        new_action = action_service.create_action(action)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Action created successfully with ID: {new_action.id}"
        )
        return new_action
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while creating action: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while creating action: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get("/actions/{action_id}", response_model=ValidationInterface.ActionRead)
def get_action(
    action_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to get action with ID: {action_id}"
    )
    action_service = ActionService()
    try:
        action = action_service.get_action(action_id)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Action retrieved successfully with ID: {action_id}"
        )
        return action
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while retrieving action {action_id}: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while retrieving action {action_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.put("/actions/{action_id}", response_model=ValidationInterface.ActionRead)
def update_action_status(
    action_id: str,
    action_update: ValidationInterface.ActionUpdate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to update status of action ID: {action_id}"
    )
    action_service = ActionService()
    try:
        updated_action = action_service.update_action_status(action_id, action_update)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Action status updated successfully for action ID: {action_id}"
        )
        return updated_action
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while updating action status {action_id}: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while updating action status {action_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get(
    "/runs/{run_id}/actions/status", response_model=List[ValidationInterface.ActionRead]
)
def get_actions_by_status(
    run_id: str,
    status: Optional[str] = "pending",
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to get actions for run ID: {run_id} with status: {status}"
    )
    action_service = ActionService()
    try:
        actions = action_service.get_actions_by_status(run_id, status)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Actions retrieved successfully for run ID: {run_id} with status: {status}"
        )
        return actions
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while retrieving actions for run {run_id} with status {status}: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while retrieving actions for run {run_id} with status {status}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get("/actions/pending/{run_id}", response_model=List[Dict[str, Any]])
def get_pending_actions(
    run_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Retrieve all pending actions with their function arguments, tool names,
    and run details. Filter by run_id.
    """
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to list pending actions for run_id: {run_id}"
    )
    action_service = ActionService()
    try:
        pending_actions = action_service.get_pending_actions(run_id)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Successfully retrieved {len(pending_actions)} pending action(s) for run_id: {run_id}."
        )
        return pending_actions
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while listing pending actions: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while listing pending actions: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.delete("/actions/{action_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_action(
    action_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"User '{auth_key.user_id}' - Received request to delete action with ID: {action_id}"
    )
    action_service = ActionService()
    try:
        action_service.delete_action(action_id)
        logging_utility.info(
            f"User '{auth_key.user_id}' - Action deleted successfully with ID: {action_id}"
        )
        return
    except HTTPException as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - HTTP error occurred while deleting action {action_id}: {str(e)}"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"User '{auth_key.user_id}' - An unexpected error occurred while deleting action {action_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )
