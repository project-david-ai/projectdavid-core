import hmac
import os
import re
import time
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator
from sandbox.services.logging_service import LoggingUtility

# Initialize logging utility
logging_utility = LoggingUtility()


class SignedUrlRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    client_ip: Optional[str] = None
    custom_expiry: Optional[int] = Field(None, ge=60, le=86400)

    @field_validator("filename")
    def validate_filename(cls, v):
        if not re.match(r"^[\w\-\.]+$", v):
            raise ValueError("Invalid filename format")
        if ".." in v or v.startswith("/"):
            raise ValueError("Path traversal detected")
        return v


class SignedUrlService:
    def __init__(
        self,
        secret_key: str,
        storage_path: str,
        max_expiry: int = 86400,
        rate_limit: int = 100,
    ):
        self.secret_key = secret_key.encode()
        self.storage_path = storage_path
        self.max_expiry = max_expiry
        self.rate_limiter = {}  # In production, use Redis instead

        # Ensure secure directory structure
        os.makedirs(self.storage_path, exist_ok=True)
        os.chmod(self.storage_path, 0o700)
        logging_utility.info(
            "SignedUrlService initialized with storage_path=%s", storage_path
        )

    def generate_signed_url(self, request: SignedUrlRequest) -> str:
        """Generate HMAC-secured URL with enhanced validation"""
        try:
            file_path = os.path.join(self.storage_path, request.filename)
            logging_utility.info("Generating signed URL for file=%s", request.filename)

            if not os.path.isfile(file_path):
                logging_utility.warning("File not found: %s", request.filename)
                raise HTTPException(404, "File not found")

            # Rate limiting
            if self.rate_limiter.get(request.client_ip, 0) >= 100:
                logging_utility.warning(
                    "Rate limit exceeded for client_ip=%s", request.client_ip
                )
                raise HTTPException(429, "Rate limit exceeded")

            # Calculate expiry time
            expiry = request.custom_expiry or self.max_expiry
            expires_at = int(time.time()) + min(expiry, self.max_expiry)
            logging_utility.debug("Expiry time calculated: %s", expires_at)

            # HMAC-based signature
            message = f"{request.filename}{expires_at}".encode()
            signature = hmac.new(
                self.secret_key, msg=message, digestmod="sha256"
            ).hexdigest()
            logging_utility.debug("Signature generated for file=%s", request.filename)

            return f"/files/{request.filename}?sig={signature}&exp={expires_at}"

        except HTTPException as e:
            logging_utility.error("HTTPException during URL generation: %s", str(e))
            raise
        except Exception as e:
            logging_utility.critical(
                "Unexpected error during URL generation: %s", str(e)
            )
            raise HTTPException(500, "Internal server error") from e

    def validate_url(self, filename: str, signature: str, expires: int) -> bool:
        """Validate URL with security checks"""
        try:
            logging_utility.info("Validating signed URL for file=%s", filename)

            # Timing-safe comparison and validation
            if time.time() > expires:
                logging_utility.warning("URL expired for file=%s", filename)
                raise HTTPException(403, "URL expired")

            file_path = os.path.join(self.storage_path, filename)
            if not os.path.isfile(file_path):
                logging_utility.warning(
                    "File not found during validation: %s", filename
                )
                raise HTTPException(404, "File not found")

            # Recompute signature
            message = f"{filename}{expires}".encode()
            expected_sig = hmac.new(
                self.secret_key, msg=message, digestmod="sha256"
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                logging_utility.error("Invalid signature for file=%s", filename)
                raise HTTPException(403, "Invalid signature")

            logging_utility.info("URL validation successful for file=%s", filename)
            return True

        except HTTPException as e:
            logging_utility.error("HTTPException during URL validation: %s", str(e))
            raise
        except Exception as e:
            logging_utility.critical(
                "Unexpected error during URL validation: %s", str(e)
            )
            raise HTTPException(500, "Internal server error") from e
