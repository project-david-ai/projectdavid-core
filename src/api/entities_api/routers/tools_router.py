# src/api/entities_api/routers/tools_router.py
from typing import List, Optional

from entities_api.services.scratchpad_service import ScratchpadService
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

# --- Core Dependencies ---
from src.api.entities_api.dependencies import (
    get_api_key,
    get_db,
    get_scratchpad_service,
    get_web_reader,
)
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import User as UserModel
from src.api.entities_api.orchestration.mixins.web_search_mixin import SearxNGClient
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.web_reader import UniversalWebReader
from src.api.entities_api.utils.check_admin_status import _is_admin

# --- Router Setup ---
router = APIRouter()
logging_utility = LoggingUtility()


# --- Request Models ---


class WebReadRequest(BaseModel):
    url: str
    force_refresh: bool = False


class WebScrollRequest(BaseModel):
    url: str
    page: int


class WebSearchRequest(BaseModel):
    url: str
    query: str


class SerpSearchRequest(BaseModel):
    query: str
    count: int = 10
    engines: Optional[List[str]] = None


class ScratchpadReadRequest(BaseModel):
    thread_id: str


class ScratchpadUpdateRequest(BaseModel):
    thread_id: str
    content: str


class ScratchpadAppendRequest(BaseModel):
    thread_id: str
    note: str


# --- Helper ---


def verify_admin_privileges(db: Session, auth_key: ApiKeyModel) -> UserModel:
    """
    Enforce admin-only access. Raises 403 if the authenticated user is not an admin.
    Returns the admin UserModel on success.
    """
    if not _is_admin(auth_key.user_id, db):
        logging_utility.warning(
            f"Unauthorized web access attempt by user ID: {auth_key.user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to use Web Tools.",
        )
    # Return the user object for logging purposes
    return db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()


# -----------------------------------------------------------------------------
# WEB TOOL ROUTES
# -----------------------------------------------------------------------------


@router.post("/tools/web/read", summary="Read a URL (Page 0)")
async def read_url(
    payload: WebReadRequest,
    reader: UniversalWebReader = Depends(get_web_reader),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    **Agent Action:** Read a new URL via browserless.
    **Security:** Admin Only.
    """
    admin_user = verify_admin_privileges(db, auth_key)
    logging_utility.info(
        f"Admin '{admin_user.email}' requesting to read URL: {payload.url}"
    )
    try:
        result = await reader.read(payload.url, force_refresh=payload.force_refresh)
        return {"content": result}
    except Exception as e:
        logging_utility.error(f"Web read failed for {payload.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Web browsing failed: {str(e)}")


@router.post("/tools/web/scroll", summary="Scroll to a specific page")
async def scroll_url(
    payload: WebScrollRequest,
    reader: UniversalWebReader = Depends(get_web_reader),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    **Agent Action:** Scroll an existing cached URL to the next page chunk.
    **Security:** Admin Only.
    """
    admin_user = verify_admin_privileges(db, auth_key)
    logging_utility.info(
        f"Admin '{admin_user.email}' requesting scroll on URL: {payload.url} (Page {payload.page})"
    )
    try:
        result = await reader.scroll(payload.url, payload.page)
        return {"content": result}
    except Exception as e:
        logging_utility.error(f"Web scroll failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scrolling failed: {str(e)}")


@router.post("/tools/web/search", summary="Search text inside a loaded URL")
async def search_url(
    payload: WebSearchRequest,
    reader: UniversalWebReader = Depends(get_web_reader),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    **Agent Action:** Search for a specific term across ALL cached pages of a URL.
    **Security:** Admin Only.
    """
    verify_admin_privileges(db, auth_key)
    try:
        result = await reader.search(payload.url, payload.query)
        return {"content": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/web/serp", summary="Structured SERP search via SearxNG")
async def serp_search(
    payload: SerpSearchRequest,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    **Agent Action:** Perform a web discovery search via the internal SearxNG container.
    **Security:** Admin Only.
    """
    admin_user = verify_admin_privileges(db, auth_key)
    logging_utility.info(
        f"Admin '{admin_user.email}' SERP search: '{payload.query}' "
        f"engines={payload.engines or 'default'} count={payload.count}"
    )
    try:
        searxng = SearxNGClient()
        result = await searxng.format_for_agent(
            query=payload.query,
            count=payload.count,
            engines=payload.engines,
        )
        return {"content": result}
    except Exception as e:
        logging_utility.error(f"SERP search failed for '{payload.query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"SERP search failed: {str(e)}")


# -----------------------------------------------------------------------------
# SCRATCHPAD TOOL ROUTES
# -----------------------------------------------------------------------------


@router.post("/tools/scratchpad/read", summary="Read the current research plan/notes")
async def read_scratchpad(
    payload: ScratchpadReadRequest,
    service: ScratchpadService = Depends(get_scratchpad_service),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """**Agent Action:** Retrieve the current state of the scratchpad."""
    verify_admin_privileges(db, auth_key)
    try:
        content = await service.get_formatted_view(payload.thread_id)
        return {"content": content}
    except Exception as e:
        logging_utility.error(f"Scratchpad read failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/scratchpad/update", summary="Overwrite the research plan")
async def update_scratchpad(
    payload: ScratchpadUpdateRequest,
    service: ScratchpadService = Depends(get_scratchpad_service),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """**Agent Action:** Rewrite the scratchpad."""
    verify_admin_privileges(db, auth_key)
    try:
        msg = await service.update_content(payload.thread_id, payload.content)
        return {"status": "success", "message": msg}
    except Exception as e:
        logging_utility.error(f"Scratchpad update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/scratchpad/append", summary="Add a note to the scratchpad")
async def append_scratchpad(
    payload: ScratchpadAppendRequest,
    service: ScratchpadService = Depends(get_scratchpad_service),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """**Agent Action:** Quick-add a finding without rewriting everything."""
    verify_admin_privileges(db, auth_key)
    try:
        msg = await service.append_note(payload.thread_id, payload.note)
        return {"status": "success", "message": msg}
    except Exception as e:
        logging_utility.error(f"Scratchpad append failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
