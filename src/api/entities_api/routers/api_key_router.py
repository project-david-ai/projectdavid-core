from fastapi import APIRouter, Depends, HTTPException, Response, status
from projectdavid_common.schemas.api_key_schemas import (ApiKeyCreateRequest,
                                                         ApiKeyCreateResponse,
                                                         ApiKeyDetails,
                                                         ApiKeyListResponse)
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.api_key_service import ApiKeyService
from src.api.entities_api.utils.check_admin_status import _is_admin

auth_router = APIRouter(tags=["API Keys"])


@auth_router.get("/auth/validate-key")
async def validate_key(auth_key: ApiKey = Depends(get_api_key)):
    """Internal endpoint for service-to-service API key validation."""
    return {"user_id": auth_key.user_id, "valid": True}


router = APIRouter(
    prefix="/users/{user_id}/apikeys",
    tags=["API Keys"],
    responses={404: {"description": "User or Key not found"}},
)


@router.get("/auth/validate-key")
async def validate_key(auth_key: ApiKey = Depends(get_api_key)):
    """Internal endpoint for service-to-service API key validation."""
    return {"user_id": auth_key.user_id, "valid": True}


def verify_user_access(
    requested_user_id: str,
    authenticated_key: ApiKeyModel,
    db: Session,
):
    """
    Allow access if:
      - The authenticated key belongs to the requested user (self-service), OR
      - The authenticated key's owner is an admin (elevated access).
    """
    if authenticated_key.user_id == requested_user_id:
        return
    if _is_admin(authenticated_key.user_id, db):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized to manage API keys for this user.",
    )


@router.post(
    "",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API Key",
    description="Generates a new API key for the specified user. The plain key is returned only once.",
)
def create_api_key(
    user_id: str,
    request_data: ApiKeyCreateRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Creates an API key for the user specified in the path (`user_id`).
    - **Authorization**: Authenticated user must match `user_id`, or be an admin.
    - **Input**: Optional key name and expiration days.
    - **Output**: The generated plain API key and its details. **Store the key immediately.**
    """
    verify_user_access(requested_user_id=user_id, authenticated_key=auth_key, db=db)
    service = ApiKeyService(db=db)
    try:
        plain_key, created_key_record = service.create_key(
            user_id=user_id,
            key_name=request_data.key_name,
            expires_in_days=request_data.expires_in_days,
        )
        key_details = ApiKeyDetails.model_validate(created_key_record)
        return ApiKeyCreateResponse(plain_key=plain_key, details=key_details)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get(
    "",
    response_model=ApiKeyListResponse,
    summary="List API Keys",
    description="Retrieves a list of API keys for the specified user.",
)
def list_api_keys(
    user_id: str,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Lists API keys for the user specified in the path (`user_id`).
    - **Authorization**: Authenticated user must match `user_id`, or be an admin.
    - **Query Param**: `include_inactive` (default False) to show revoked keys.
    - **Output**: A list of API key details (prefix, name, dates, status).
    """
    verify_user_access(requested_user_id=user_id, authenticated_key=auth_key, db=db)
    service = ApiKeyService(db=db)
    try:
        keys_orm = service.list_keys_for_user(user_id=user_id, include_inactive=include_inactive)
        keys_details = [ApiKeyDetails.model_validate(key) for key in keys_orm]
        return ApiKeyListResponse(keys=keys_details)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}",
        )


@router.get(
    "/{key_prefix}",
    response_model=ApiKeyDetails,
    summary="Get API Key Details",
    description="Retrieves details for a specific API key using its prefix.",
)
def get_api_key_details(
    user_id: str,
    key_prefix: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Gets details for a specific API key identified by its `key_prefix` for the given `user_id`.
    - **Authorization**: Authenticated user must match `user_id`, or be an admin.
    - **Output**: API key details (prefix, name, dates, status).
    """
    verify_user_access(requested_user_id=user_id, authenticated_key=auth_key, db=db)
    service = ApiKeyService(db=db)
    try:
        key_orm = service.get_key_details_by_prefix(user_id=user_id, key_prefix=key_prefix)
        if not key_orm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API Key with prefix '{key_prefix}' not found for this user.",
            )
        return ApiKeyDetails.model_validate(key_orm)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API key details: {str(e)}",
        )


@router.delete(
    "/{key_prefix}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API Key",
    description="Revokes (deactivates) a specific API key using its prefix.",
)
def revoke_api_key(
    user_id: str,
    key_prefix: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Revokes (sets `is_active=False`) an API key identified by its `key_prefix` for the given `user_id`.
    - **Authorization**: Authenticated user must match `user_id`, or be an admin.
    - **Output**: `204 No Content` on success.
    """
    verify_user_access(requested_user_id=user_id, authenticated_key=auth_key, db=db)
    service = ApiKeyService(db=db)
    try:
        revoked = service.revoke_key(user_id=user_id, key_prefix=key_prefix)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API Key with prefix '{key_prefix}' not found for this user or could not be revoked.",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}",
        )
