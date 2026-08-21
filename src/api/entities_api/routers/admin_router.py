from fastapi import APIRouter, Depends, HTTPException, status
from projectdavid_common import ValidationInterface
from projectdavid_common.schemas.api_key_schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyDetails,
)
from projectdavid_common.utilities.logging_service import LoggingUtility
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import User as UserModel
from src.api.entities_api.services.admin_provisioning_service import (
    AdminProvisioningService,
)
from src.api.entities_api.services.api_key_service import ApiKeyService

admin_router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    responses={
        403: {"description": "Admin privileges required"},
        401: {"description": "Authentication required / Invalid API Key"},
    },
)
logging_utility = LoggingUtility()
validation = ValidationInterface()


class AdminProvisionUserRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    full_name: str = Field(min_length=1, max_length=255)
    external_reference: str = Field(min_length=3, max_length=255)
    key_name: str = Field(default="project-black-bird", min_length=1, max_length=100)


class AdminProvisionUserResponse(BaseModel):
    user: validation.UserRead
    key: ApiKeyCreateResponse
    user_created: bool
    keys_revoked: int


@admin_router.post(
    "/users/provision",
    response_model=AdminProvisionUserResponse,
    status_code=status.HTTP_200_OK,
    summary="Admin: Ensure User and API Key",
)
def admin_provision_user(
    request_data: AdminProvisionUserRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """Ensure one verified Project David user and one active Black Bird key."""
    requesting_user = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_user or not requesting_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation.",
        )

    result = AdminProvisioningService(db).ensure_user_api_key(
        email=str(request_data.email),
        full_name=request_data.full_name,
        external_reference=request_data.external_reference,
        key_name=request_data.key_name,
    )
    logging_utility.info(
        "Admin %s ensured Project David access for user %s (prefix=%s, revoked=%d)",
        requesting_user.id,
        result.user.id,
        result.api_key.prefix,
        result.keys_revoked,
    )
    return AdminProvisionUserResponse(
        user=validation.UserRead.model_validate(result.user),
        key=ApiKeyCreateResponse(
            plain_key=result.plain_key,
            details=ApiKeyDetails.model_validate(result.api_key),
        ),
        user_created=result.user_created,
        keys_revoked=result.keys_revoked,
    )


@admin_router.post(
    "/users/{target_user_id}/keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Admin: Create API Key for User",
    description="Allows an authenticated administrator to create a new API key for any specified user.",
)
def admin_create_api_key_for_user(
    target_user_id: str,
    request_data: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Admin Only: Creates an API key for the user specified in the path (`target_user_id`).

    - **Authentication**: Requires a valid Admin API key in the `X-API-Key` header.
    - **Authorization**: The user associated with the authentication key must have `is_admin=True`.
    - **Input**: Target user ID in path, optional key details in body.
    - **Output**: The generated plain API key and its details. **Store the key immediately.**
    """
    logging_utility.info(
        f"Admin request received from user {auth_key.user_id} (Key Prefix: {auth_key.prefix}) to create key for user {target_user_id}."
    )
    requesting_user = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_user or not requesting_user.is_admin:
        logging_utility.warning(
            f"Authorization Failed: User {auth_key.user_id} attempted admin operation without admin rights."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation.",
        )
    logging_utility.info(
        f"Admin user {requesting_user.id} authorized. Proceeding to create key for user {target_user_id}."
    )
    service = ApiKeyService(db=db)
    try:
        plain_key, created_key_record = service.create_key(
            user_id=target_user_id,
            key_name=request_data.key_name,
            expires_in_days=request_data.expires_in_days,
        )
        key_details = ApiKeyDetails.model_validate(created_key_record)
        logging_utility.info(
            f"API Key (Prefix: {key_details.prefix}) created successfully for user {target_user_id} by admin {requesting_user.id}."
        )
        return ApiKeyCreateResponse(plain_key=plain_key, details=key_details)
    except HTTPException as e:
        logging_utility.error(
            f"HTTP error during admin key creation for user {target_user_id}: {e.detail} (Status: {e.status_code})"
        )
        raise e
    except Exception as e:
        logging_utility.error(
            f"Unexpected error during admin key creation for user {target_user_id}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Admin failed to create API key for user {target_user_id}: An internal error occurred.",
        )
