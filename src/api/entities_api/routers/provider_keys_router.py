"""Authenticated, tenant-owned inference provider key management API."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from projectdavid_common import ValidationInterface

from src.api.entities_api.dependencies import get_api_key
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
)
from src.api.entities_api.services.provider_key_service import (
    ProviderKeyConfigurationError,
    ProviderKeyNotFoundError,
    ProviderKeyService,
    ProviderKeyUnavailableError,
)

validator = ValidationInterface()
router = APIRouter(prefix="/provider-keys", tags=["Provider Keys"])


def get_provider_key_service() -> ProviderKeyService:
    return ProviderKeyService()


@router.post(
    "",
    response_model=validator.ProviderKeyRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Existing active registration returned"},
        503: {"description": "Credential security configuration unavailable"},
    },
)
def register_provider_key(
    registration: validator.ProviderKeyCreate,
    response: Response,
    auth_key: ApiKeyModel = Depends(get_api_key),
    service: ProviderKeyService = Depends(get_provider_key_service),
):
    try:
        result, created = service.register(registration, user_id=auth_key.user_id)
    except (ProviderKeyConfigurationError, CredentialConfigurationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider credential security configuration unavailable",
        ) from exc

    if not created:
        response.status_code = status.HTTP_200_OK

    return result


@router.get("", response_model=list[validator.ProviderKeyRead])
def list_provider_keys(
    auth_key: ApiKeyModel = Depends(get_api_key),
    service: ProviderKeyService = Depends(get_provider_key_service),
):
    return service.list_keys(user_id=auth_key.user_id)


@router.get("/{provider_key_id}", response_model=validator.ProviderKeyRead)
def get_provider_key(
    provider_key_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
    service: ProviderKeyService = Depends(get_provider_key_service),
):
    try:
        return service.get_key(
            provider_key_id=provider_key_id,
            user_id=auth_key.user_id,
        )
    except ProviderKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider key not found",
        ) from exc


@router.patch("/{provider_key_id}", response_model=validator.ProviderKeyRead)
def update_provider_key(
    provider_key_id: str,
    update: validator.ProviderKeyUpdate,
    auth_key: ApiKeyModel = Depends(get_api_key),
    service: ProviderKeyService = Depends(get_provider_key_service),
):
    try:
        return service.update_key(
            provider_key_id=provider_key_id,
            user_id=auth_key.user_id,
            update=update,
        )
    except ProviderKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider key not found",
        ) from exc


@router.delete(
    "/{provider_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def revoke_provider_key(
    provider_key_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
    service: ProviderKeyService = Depends(get_provider_key_service),
):
    try:
        service.revoke(
            provider_key_id=provider_key_id,
            user_id=auth_key.user_id,
        )
    except ProviderKeyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Provider key not found",
        ) from exc
    except ProviderKeyUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Provider credential cannot be revoked",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
