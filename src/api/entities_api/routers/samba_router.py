import hashlib
import hmac
import os
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from projectdavid_common import ValidationInterface
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.file_service import FileService
from src.api.entities_api.services.logging_service import LoggingUtility

router = APIRouter()
logging_utility = LoggingUtility()


@router.post(
    "/uploads", response_model=ValidationInterface.FileResponse, status_code=201
)
def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Uploading file: {file.filename}")
    request = ValidationInterface.FileUploadRequest(purpose=purpose, user_id=user_id)
    file_service = FileService(db)
    file_metadata = file_service.upload_file(file, request)
    return ValidationInterface.FileResponse.model_validate(file_metadata)


@router.get("/uploads/{file_id}", response_model=ValidationInterface.FileResponse)
def get_file_by_id(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Retrieving file metadata for ID: {file_id}"
    )
    file_service = FileService(db)
    file_metadata = file_service.get_file_by_id(file_id)
    if not file_metadata:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
    return ValidationInterface.FileResponse.model_validate(file_metadata)


@router.get("/uploads/{file_id}/object", response_class=Response)
def download_file_as_object(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    file_service = FileService(db)
    file_obj = file_service.get_file_as_object(file_id)
    return Response(content=file_obj.read(), media_type="application/octet-stream")


@router.get("/uploads/{file_id}/signed-url", response_model=dict)
def get_signed_url(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    file_service = FileService(db)
    signed_url = file_service.get_file_as_signed_url(file_id)
    return {"signed_url": signed_url}


@router.get("/uploads/{file_id}/base64", response_model=dict)
def get_file_as_base64(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    file_service = FileService(db)
    b64_str = file_service.get_file_as_base64(file_id)
    return {"base64": b64_str}


@router.delete("/uploads/{file_id}", response_model=dict)
def delete_file(
    file_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    file_service = FileService(db)
    success = file_service.delete_file_by_id(file_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"File with ID {file_id} not found")
    return {"deleted": True}
