"""
Project David — Permanent File Retention Test
---------------------------------------------
1. Upload a file through the authenticated Core API.
2. Retrieve its metadata.
3. Inspect the persisted database record.
4. Assert that expires_at is NULL.
5. Hard-delete the test file.

No Agents.
No Runs.
No inference.
No provider API keys.

Uses the existing Core test-user key from .tests.env.

Run from the projectdavid-core repository root:

    python tests/test_file_retention.py
"""

import os
import subprocess
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ENV = PROJECT_ROOT / "tests" / "integration" / ".tests.env"

load_dotenv(TEST_ENV)

CORE_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

BASE_URL = (
    os.getenv(
        "FILE_RETENTION_TEST_BASE_URL",
        os.getenv("ENTITIES_BASE_URL", "http://localhost:80"),
    )
    .rstrip("/")
    .removesuffix("/v1")
)

DOCKER_CONTEXT = "desktop-linux"
CORE_CONTAINER = "fastapi_cosmic_catalyst"

if not CORE_API_KEY:
    raise RuntimeError("Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY " "in .tests.env")


# ------------------------------------------------------------
# Database verification
# ------------------------------------------------------------

DB_CHECK = """
import sys

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import File

file_id = sys.argv[1]

db = SessionLocal()

try:
    record = (
        db.query(File)
        .filter(File.id == file_id)
        .one_or_none()
    )

    if record is None:
        raise RuntimeError(
            "Uploaded file not found in Core database"
        )

    print("DATABASE RECORD: FOUND")
    print("EXPIRES AT:", record.expires_at)

    assert record.expires_at is None, (
        "FAIL: File has an automatic expiry timestamp"
    )

    print("PERMANENT RETENTION: PASS")

finally:
    db.close()
"""


def verify_database(file_id: str) -> None:
    """Inspect the persisted file in the running Core container."""

    result = subprocess.run(
        [
            "docker",
            "--context",
            DOCKER_CONTEXT,
            "exec",
            CORE_CONTAINER,
            "python",
            "-c",
            DB_CHECK,
            file_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())

        raise RuntimeError("Database retention verification failed")


# ------------------------------------------------------------
# Integration test
# ------------------------------------------------------------


def main() -> None:
    print("\nPROJECT DAVID — FILE RETENTION TEST")
    print("-" * 55)
    print("Core:", BASE_URL)
    print("Container:", CORE_CONTAINER)

    filename = f"projectdavid-retention-" f"{uuid.uuid4().hex[:12]}.txt"

    content = (
        b"Project David permanent file retention test.\n"
        b"This file must not receive an automatic expiry.\n"
    )

    headers = {
        "X-API-Key": CORE_API_KEY,
    }

    file_id = None
    passed = False

    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
    ) as client:

        try:
            # ------------------------------------------------
            # 1. Upload
            # ------------------------------------------------

            print("\n1. UPLOAD")

            response = client.post(
                f"{BASE_URL}/v1/uploads",
                headers=headers,
                data={"purpose": "assistants"},
                files={
                    "file": (
                        filename,
                        content,
                        "text/plain",
                    )
                },
            )

            response.raise_for_status()

            uploaded = response.json()
            file_id = uploaded["id"]

            assert uploaded["filename"] == filename

            print("UPLOAD: PASS")
            print("File ID:", file_id)
            print("Filename:", filename)

            # ------------------------------------------------
            # 2. Retrieve metadata
            # ------------------------------------------------

            print("\n2. METADATA RETRIEVAL")

            response = client.get(
                f"{BASE_URL}/v1/files/{file_id}",
                headers=headers,
            )

            response.raise_for_status()

            metadata = response.json()

            assert metadata["id"] == file_id
            assert metadata["filename"] == filename
            assert metadata["purpose"] == "assistants"

            print("METADATA RETRIEVAL: PASS")

            # ------------------------------------------------
            # 3. Verify actual database retention
            # ------------------------------------------------

            print("\n3. DATABASE RETENTION")

            verify_database(file_id)

            passed = True

        finally:
            # ------------------------------------------------
            # 4. Explicit test-file cleanup
            # ------------------------------------------------

            if file_id:
                print("\n4. TEST FILE CLEANUP")

                response = client.delete(
                    f"{BASE_URL}/v1/files/{file_id}",
                    headers=headers,
                )

                response.raise_for_status()

                result = response.json()

                assert result.get("deleted") is True, "Core did not confirm deletion"

                print("HARD DELETE: PASS")

    if passed:
        print("\n" + "=" * 55)
        print("FILE RETENTION INTEGRATION TEST: PASSED")
        print("=" * 55)


if __name__ == "__main__":
    main()
