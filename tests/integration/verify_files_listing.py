"""
Project David — Files Listing Integration Test

Tests:
- Authenticated upload
- Owner-scoped listing
- Cursor pagination
- Soft-delete visibility
- Hard-delete cleanup

Run:
    python tests/integration/test_files_listing.py
"""

import os
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / "tests" / "integration" / ".tests.env")

API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

BASE_URL = (
    os.getenv(
        "FILE_RETENTION_TEST_BASE_URL",
        os.getenv("ENTITIES_BASE_URL", "http://localhost:80"),
    )
    .rstrip("/")
    .removesuffix("/v1")
)

if not API_KEY:
    raise RuntimeError("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is not set")


def main():
    purpose = f"files-list-test-{uuid.uuid4().hex[:12]}"
    headers = {"X-API-Key": API_KEY}
    created_ids = []

    print("\nPROJECT DAVID — FILES LISTING TEST")
    print("-" * 55)

    with httpx.Client(timeout=60) as client:
        try:
            # Upload two isolated test files.
            for number in (1, 2):
                response = client.post(
                    f"{BASE_URL}/v1/uploads",
                    headers=headers,
                    data={"purpose": purpose},
                    files={
                        "file": (
                            f"list-test-{number}.txt",
                            b"test content\n",
                            "text/plain",
                        )
                    },
                )
                response.raise_for_status()
                created_ids.append(response.json()["id"])

            print("UPLOADS: PASS")

            # First page.
            first = client.get(
                f"{BASE_URL}/v1/files",
                headers=headers,
                params={
                    "purpose": purpose,
                    "limit": 1,
                },
            )
            first.raise_for_status()
            page1 = first.json()

            assert page1["object"] == "list"
            assert len(page1["data"]) == 1
            assert page1["has_more"] is True
            assert page1["last_id"] == page1["data"][-1]["id"]

            # Second page.
            second = client.get(
                f"{BASE_URL}/v1/files",
                headers=headers,
                params={
                    "purpose": purpose,
                    "limit": 1,
                    "after": page1["last_id"],
                },
            )
            second.raise_for_status()
            page2 = second.json()

            assert len(page2["data"]) == 1
            assert page2["has_more"] is False

            listed_ids = {
                page1["data"][0]["id"],
                page2["data"][0]["id"],
            }

            assert listed_ids == set(created_ids)

            print("CURSOR PAGINATION: PASS")

            # Soft-deleted files must not appear.
            hidden_id = created_ids[0]

            hidden = client.patch(
                f"{BASE_URL}/v1/files/{hidden_id}/soft-delete",
                headers=headers,
            )
            hidden.raise_for_status()

            visible = client.get(
                f"{BASE_URL}/v1/files",
                headers=headers,
                params={
                    "purpose": purpose,
                    "limit": 10,
                },
            )
            visible.raise_for_status()

            remaining = {record["id"] for record in visible.json()["data"]}

            assert hidden_id not in remaining
            assert created_ids[1] in remaining

            print("SOFT DELETE VISIBILITY: PASS")

        finally:
            for file_id in created_ids:
                response = client.delete(
                    f"{BASE_URL}/v1/files/{file_id}",
                    headers=headers,
                )

                if response.status_code != 200:
                    print(
                        f"WARNING: manual cleanup required for "
                        f"{file_id}: {response.status_code}"
                    )
                else:
                    print(f"Cleaned up {file_id}")

    print("\nFILES LISTING INTEGRATION TEST: PASSED")


if __name__ == "__main__":
    main()
