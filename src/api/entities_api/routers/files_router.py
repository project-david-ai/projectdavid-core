import hashlib
import hmac
import os
from datetime import datetime
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import FileDeleteResponse, FileResponse
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import File as FileModel
from src.api.entities_api.models.models import User
from src.api.entities_api.services.erasure_state import erasure_state
from src.api.entities_api.services.file_service import FileService
from src.api.entities_api.services.request_activity import track_authenticated_request

load_dotenv()
router = APIRouter()
logging_utility = LoggingUtility()


# ----------------------------------------------------------------------
# Signed URL download
# ----------------------------------------------------------------------


@router.get(
    "/files/download",
    include_in_schema=False,
    response_class=StreamingResponse,
    summary="Download file via signed URL (no API key required)",
)
def download_file(
    file_id: str = Query(...),
    expires: int = Query(...),
    signature: str = Query(...),
    use_real_filename: bool = Query(False),
    db: Session = Depends(get_db),
):
    log_msg = f"!!!!!! Entered /files/download route for file_id: {file_id} !!!!!!"
    print(log_msg, flush=True)
    logging_utility.error(log_msg)

    if datetime.utcnow().timestamp() > expires:
        logging_utility.warning(f"Download URL expired for file_id: {file_id}")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Signed URL expired")

    if not _verify_sig(file_id, expires, signature, use_real_filename):
        logging_utility.warning(f"Invalid signature for file_id: {file_id}")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature")

    logging_utility.info(f"Signature OK, proceeding to stream file_id: {file_id}")

    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()

    if not file_record or not file_record.user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")

    # A signed URL is a bearer credential, even though it bypasses API-key
    # authentication. Register its owner before the fresh receipt read, so
    # admitted streams remain visible to the account-erasure coordinator.
    owner_id = file_record.user_id
    track_authenticated_request(owner_id)
    with SessionLocal() as admission_db:
        if erasure_state(admission_db, owner_id) is not None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")

    svc = FileService(db)
    stream, orig_name, mime = svc.get_file_with_metadata(
        file_id,
        user_id=owner_id,
    )

    fname = orig_name if use_real_filename else file_id
    disp = (
        "inline"
        if mime and mime.startswith(("image/", "text/", "application/pdf"))
        else "attachment"
    )

    return StreamingResponse(
        stream,
        media_type=mime or "application/octet-stream",
        headers={
            "Content-Disposition": f'{disp}; filename="{fname}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------


@router.post(
    "/uploads",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
)
def upload_file_endpoint(
    purpose: str = Form(..., description="Purpose (e.g. assistants)"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
) -> FileResponse:
    user_id = auth_key.user_id

    logging_utility.info(
        "User %s uploading %s (%s)",
        user_id,
        file.filename,
        purpose,
    )

    service = FileService(db)

    created = service.upload_file(
        file=file,
        request=type(
            "Req",
            (),
            {
                "purpose": purpose,
                "user_id": user_id,
            },
        ),
    )

    return FileResponse.model_validate(created)


@router.post("/internal/sandbox/artifacts", status_code=201, include_in_schema=False)
def upload_sandbox_artifact(
    owner_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """Trusted sandbox service delegates ownership, not the socket client."""
    admin = db.get(User, auth_key.user_id)
    if admin is None or not admin.is_admin:
        raise HTTPException(403, "Admin privileges required.")
    track_authenticated_request(owner_id)
    with SessionLocal() as admission_db:
        if (
            erasure_state(admission_db, owner_id) is not None
            or admission_db.get(User, owner_id) is None
        ):
            raise HTTPException(403, "Sandbox access denied.")
    # Fail before persistence if download signing is misconfigured.
    if not os.getenv("SIGNED_URL_SECRET") or not os.getenv("DOWNLOAD_BASE_URL"):
        raise HTTPException(503, "Download signing unavailable.")
    created = FileService(db).upload_file(
        file=file,
        request=type("Req", (), {"purpose": "assistants", "user_id": owner_id}),
    )
    result = FileResponse.model_validate(created)
    return {
        "id": result.id,
        "filename": result.filename,
        "url": _build_signed_url(result.id, 600, True),
    }


# ----------------------------------------------------------------------
# List files — NEW
#
# Authenticated, owner-scoped, soft-deleted files excluded.
#
# Ordering: created_at DESC, id DESC
# Cursor: file ID from the previous page.
# ----------------------------------------------------------------------


@router.get(
    "/files",
    response_model=dict,
    summary="List files belonging to the authenticated user",
)
def list_files_endpoint(
    limit: int = Query(50, ge=1, le=100),
    after: str | None = Query(
        None,
        description="Last file ID from the previous page",
    ),
    purpose: str | None = Query(None),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    scope = (
        FileModel.user_id == auth_key.user_id,
        FileModel.deleted_at.is_(None),
    )

    query = db.query(FileModel).filter(*scope)

    if purpose is not None:
        query = query.filter(FileModel.purpose == purpose)

    if after:
        anchor = (
            db.query(FileModel)
            .filter(
                FileModel.id == after,
                *scope,
            )
            .first()
        )

        if anchor is None or (purpose is not None and anchor.purpose != purpose):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "File cursor not found",
            )

        query = query.filter(
            or_(
                FileModel.created_at < anchor.created_at,
                and_(
                    FileModel.created_at == anchor.created_at,
                    FileModel.id < anchor.id,
                ),
            )
        )

    records = (
        query.order_by(
            FileModel.created_at.desc(),
            FileModel.id.desc(),
        )
        .limit(limit + 1)
        .all()
    )

    has_more = len(records) > limit
    page = records[:limit]

    return {
        "object": "list",
        "data": [
            FileResponse.model_validate(file).model_dump(mode="json") for file in page
        ],
        "first_id": page[0].id if page else None,
        "last_id": page[-1].id if page else None,
        "has_more": has_more,
    }


# ----------------------------------------------------------------------
# Retrieve metadata
# ----------------------------------------------------------------------


@router.get(
    "/files/{file_id}",
    response_model=FileResponse,
    summary="Retrieve file metadata",
)
def retrieve_file_metadata(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id

    logging_utility.info(
        "User %s reading metadata for %s",
        user_id,
        file_id,
    )

    data = FileService(db).get_file_by_id(
        file_id,
        user_id=user_id,
    )

    if not data:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "File not found",
        )

    return FileResponse.model_validate(data)


# ----------------------------------------------------------------------
# Hard delete
# ----------------------------------------------------------------------


@router.delete(
    "/files/{file_id}",
    response_model=FileDeleteResponse,
    summary="Hard-delete a file (removes DB row and Samba bytes)",
)
def delete_file_endpoint(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id

    logging_utility.info(
        "User %s hard-deleting %s",
        user_id,
        file_id,
    )

    if not FileService(db).delete_file_by_id(
        file_id,
        user_id=user_id,
    ):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "File not found",
        )

    return FileDeleteResponse(
        id=file_id,
        object="file",
        deleted=True,
    )


# ----------------------------------------------------------------------
# Soft delete
# ----------------------------------------------------------------------


@router.patch(
    "/files/{file_id}/soft-delete",
    response_model=FileDeleteResponse,
    summary="Soft-delete a file (preserves Samba bytes, hides from normal queries)",
)
def soft_delete_file_endpoint(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id

    logging_utility.info(
        "User %s soft-deleting %s",
        user_id,
        file_id,
    )

    if not FileService(db).soft_delete_file_by_id(
        file_id,
        user_id=user_id,
    ):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "File not found or already deleted",
        )

    return FileDeleteResponse(
        id=file_id,
        object="file",
        deleted=True,
    )


# ----------------------------------------------------------------------
# Signed URL generation
# ----------------------------------------------------------------------


@router.get(
    "/files/{file_id}/signed-url",
    response_model=dict,
    summary="Generate a temporary signed download URL",
)
def generate_signed_url(
    file_id: str,
    expires_in: int = Query(
        600,
        description="Seconds until link expires",
    ),
    use_real_filename: bool = Query(False),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id

    file_record = (
        db.query(FileModel)
        .filter(
            FileModel.id == file_id,
            FileModel.deleted_at.is_(None),
        )
        .first()
    )

    if not file_record:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "File not found",
        )

    if file_record.user_id != user_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You do not have permission to access this file.",
        )

    url = _build_signed_url(file_id, expires_in, use_real_filename)
    logging_utility.info("User %s generated signed URL for %s", user_id, file_id)
    return {"signed_url": url}


def _build_signed_url(file_id: str, expires_in: int, use_real_filename: bool) -> str:
    secret = os.getenv("SIGNED_URL_SECRET", "")
    base_url = os.getenv("DOWNLOAD_BASE_URL", "").rstrip("/")

    if not secret or not base_url:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "SIGNED_URL_SECRET or DOWNLOAD_BASE_URL not configured",
        )

    exp = int(datetime.utcnow().timestamp() + expires_in)
    payload = f"{file_id}:{exp}:{str(use_real_filename).lower()}"

    sig = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    query = urlencode(
        {
            "file_id": file_id,
            "expires": exp,
            "signature": sig,
            "use_real_filename": use_real_filename,
        }
    )

    target_path = "/v1/files/download"

    url = (
        f"{base_url}?{query}"
        if base_url.endswith(target_path)
        else f"{base_url}{target_path}?{query}"
    )

    return url


# ----------------------------------------------------------------------
# Base64 content
# ----------------------------------------------------------------------


@router.get(
    "/files/{file_id}/base64",
    response_model=dict,
    summary="Get file as Base64",
)
def get_file_as_base64(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id

    logging_utility.info(
        "User %s requesting base64 for %s",
        user_id,
        file_id,
    )

    b64 = FileService(db).get_file_as_base64(
        file_id,
        user_id=user_id,
    )

    return {
        "file_id": file_id,
        "base64": b64,
    }


# ----------------------------------------------------------------------
# Signature verification
# ----------------------------------------------------------------------


def _verify_sig(
    file_id: str,
    exp: int,
    sig: str,
    real_name: bool,
) -> bool:
    secret = os.getenv("SIGNED_URL_SECRET", "")

    if not secret:
        logging_utility.critical("SIGNED_URL_SECRET is not configured!")
        return False

    payload = f"{file_id}:{exp}:{str(real_name).lower()}"

    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, sig)
