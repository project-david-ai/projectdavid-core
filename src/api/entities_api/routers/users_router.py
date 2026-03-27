# src/api/entities_api/routers/users_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import User as UserModel
from src.api.entities_api.serializers import UserUpdate
from src.api.entities_api.services.user_service import UserService

validation = ValidationInterface()
router = APIRouter(prefix="/users", tags=["Users"])
logging_utility = LoggingUtility()


@router.post(
    "", response_model=validation.UserRead, status_code=status.HTTP_201_CREATED
)
def create_user(
    user_data: validation.UserCreate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """Creates a new user. Requires Admin authentication via API Key."""
    logging_utility.info(f"User '{auth_key.user_id}' requesting to create a new user.")
    requesting_admin = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_admin or not requesting_admin.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to create users.",
        )
    user_service = UserService()
    try:
        new_user = user_service.create_user(user_data)
        logging_utility.info(
            f"User '{new_user.email}' (ID: {new_user.id}) created successfully by admin {requesting_admin.id}."
        )
        return new_user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating the user.",
        )


@router.get("/{user_id}", response_model=validation.UserRead)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """Retrieves details for a specific user."""
    logging_utility.info(
        f"User '{auth_key.user_id}' requesting details for user ID: {user_id}"
    )
    requesting_user = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    if not requesting_user.is_admin and requesting_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to access this user's details.",
        )
    user_service = UserService()
    try:
        return user_service.get_user(user_id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.put("/{user_id}", response_model=validation.UserRead)
def update_user(
    user_id: str,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """Updates details for a specific user."""
    logging_utility.info(
        f"User '{auth_key.user_id}' requesting to update user ID: {user_id}"
    )
    requesting_user = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    is_admin_request = requesting_user.is_admin
    is_self_request = requesting_user.id == user_id
    if not is_admin_request and not is_self_request:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to update this user.",
        )
    update_data = user_update.model_dump(exclude_unset=True)
    if not is_admin_request and "is_admin" in update_data:
        del update_data["is_admin"]
    user_service = UserService()
    try:
        return user_service.update_user(user_id, UserUpdate(**update_data))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    GDPR right-to-erasure.  Permanently deletes a user and all of their data,
    including physical files from Samba and Qdrant vector store collections.
    Requires Admin authentication.
    """
    logging_utility.info(
        f"User '{auth_key.user_id}' requesting GDPR erasure for user ID: {user_id}"
    )
    requesting_admin = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_admin or not requesting_admin.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to delete users.",
        )
    if requesting_admin.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin self-deletion via API is disallowed.",
        )
    user_service = UserService()
    try:
        # erase_user() handles physical assets + messages + audit log
        # before delegating to DB cascades for the rest.
        user_service.erase_user(user_id)
        return None
    except HTTPException:
        raise
    except Exception as exc:
        logging_utility.error(
            f"Unexpected error during GDPR erasure of user {user_id}: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during user erasure.",
        )
