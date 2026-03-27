from fastapi import APIRouter, Depends, HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.threads_service import ThreadService

router = APIRouter(
    prefix="/threads",
    tags=["Threads"],
    responses={404: {"description": "Thread not found"}},
)
validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()


@router.post("", response_model=ValidationInterface.ThreadReadDetailed)
def create_thread(
    thread: ValidationInterface.ThreadCreate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Creating thread")
    participant_ids = thread.participant_ids or [auth_key.user_id]
    thread_in = ValidationInterface.ThreadCreate(
        participant_ids=participant_ids, meta_data=thread.meta_data
    )
    try:
        return ThreadService().create_thread(
            thread_in, user_id=auth_key.user_id
        )  # ← FIXED
    except Exception as e:
        logging_utility.error(f"Error creating thread: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thread")


@router.get("/{thread_id}", response_model=ValidationInterface.ThreadReadDetailed)
def get_thread(
    thread_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Fetching thread: {thread_id}")
    return ThreadService().get_thread(thread_id)  # ← read-only, no user_id needed


@router.delete("/{thread_id}", response_model=ValidationInterface.ThreadDeleted)
def delete_thread(
    thread_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Deleting thread: {thread_id}")
    return ThreadService().delete_thread(thread_id, user_id=auth_key.user_id)  # ← FIXED


@router.get("/user/{user_id}", response_model=list[str])
def list_user_threads(
    user_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Listing threads for user {user_id}")
    return ThreadService().list_threads_by_user(
        user_id
    )  # ← read-only, no user_id needed


@router.put(
    "/{thread_id}/metadata", response_model=ValidationInterface.ThreadReadDetailed
)
def update_thread_metadata(
    thread_id: str,
    metadata: dict,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Updating metadata for thread {thread_id}"
    )
    return ThreadService().update_thread_metadata(  # ← FIXED
        thread_id, metadata, user_id=auth_key.user_id
    )


@router.put("/{thread_id}", response_model=ValidationInterface.ThreadReadDetailed)
def update_thread(
    thread_id: str,
    thread_update: ValidationInterface.ThreadUpdate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Updating thread {thread_id}")
    return ThreadService().update_thread(  # ← FIXED
        thread_id, thread_update, user_id=auth_key.user_id
    )
