from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from projectdavid_common import ValidationInterface

from src.api.entities_api.dependencies import get_api_key, get_scratchpad_service
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.scratchpad_service import ScratchpadService

router = APIRouter()
logging_utility = LoggingUtility()


def _internal_error(
    operation: str,
    exc: Exception,
) -> HTTPException:
    logging_utility.error(
        "Scratchpad %s failed: %s",
        operation,
        exc,
        exc_info=True,
    )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )


@router.post(
    "/scratchpads",
    response_model=ValidationInterface.ScratchpadRead,
)
def create_scratchpad(
    scratchpad: ValidationInterface.ScratchpadCreate,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return service.create_scratchpad(
            scratchpad,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "create",
            exc,
        ) from exc


@router.get(
    "/scratchpads",
    response_model=ValidationInterface.ScratchpadList,
)
def list_scratchpads(
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return service.list_scratchpads(
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "list",
            exc,
        ) from exc


@router.get(
    "/scratchpads/{scratchpad_id}",
    response_model=ValidationInterface.ScratchpadRead,
)
def retrieve_scratchpad(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return service.retrieve_scratchpad(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "retrieve",
            exc,
        ) from exc


@router.get(
    "/threads/{thread_id}/scratchpad",
    response_model=ValidationInterface.ScratchpadRead,
)
def retrieve_scratchpad_by_thread(
    thread_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return service.retrieve_scratchpad_by_thread(
            thread_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "retrieve-by-thread",
            exc,
        ) from exc


@router.patch(
    "/scratchpads/{scratchpad_id}",
    response_model=ValidationInterface.ScratchpadRead,
)
def update_scratchpad(
    scratchpad_id: str,
    scratchpad: ValidationInterface.ScratchpadUpdate,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return service.update_metadata(
            scratchpad_id,
            scratchpad,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "update",
            exc,
        ) from exc


@router.delete(
    "/scratchpads/{scratchpad_id}",
    response_model=ValidationInterface.ScratchpadDeleted,
)
async def delete_scratchpad(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.delete_scratchpad(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "delete",
            exc,
        ) from exc


@router.get(
    "/scratchpads/{scratchpad_id}/content",
    response_model=ValidationInterface.ScratchpadContentRead,
)
async def get_scratchpad_content(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.get_content(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "get-content",
            exc,
        ) from exc


@router.put(
    "/scratchpads/{scratchpad_id}/content",
    response_model=ValidationInterface.ScratchpadContentRead,
)
async def set_scratchpad_content(
    scratchpad_id: str,
    payload: ValidationInterface.ScratchpadContentUpdate,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.set_content(
            scratchpad_id,
            payload.content,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "set-content",
            exc,
        ) from exc


@router.delete(
    "/scratchpads/{scratchpad_id}/content",
    response_model=ValidationInterface.ScratchpadStateCleared,
)
async def clear_scratchpad_content(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.clear_content(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "clear-content",
            exc,
        ) from exc


@router.post(
    "/scratchpads/{scratchpad_id}/entries",
    response_model=ValidationInterface.ScratchpadEntryRead,
)
async def append_scratchpad_entry(
    scratchpad_id: str,
    payload: ValidationInterface.ScratchpadEntryCreate,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.append_entry(
            scratchpad_id,
            payload.content,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "append-entry",
            exc,
        ) from exc


@router.get(
    "/scratchpads/{scratchpad_id}/entries",
    response_model=ValidationInterface.ScratchpadEntryList,
)
async def list_scratchpad_entries(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.list_entries(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "list-entries",
            exc,
        ) from exc


@router.delete(
    "/scratchpads/{scratchpad_id}/entries",
    response_model=ValidationInterface.ScratchpadStateCleared,
)
async def clear_scratchpad_entries(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.clear_entries(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "clear-entries",
            exc,
        ) from exc


@router.post(
    "/scratchpads/{scratchpad_id}/clear",
    response_model=ValidationInterface.ScratchpadStateCleared,
)
async def clear_scratchpad(
    scratchpad_id: str,
    service: ScratchpadService = Depends(get_scratchpad_service),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        return await service.clear_scratchpad(
            scratchpad_id,
            user_id=auth_key.user_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise _internal_error(
            "clear",
            exc,
        ) from exc


__all__ = ["router"]
