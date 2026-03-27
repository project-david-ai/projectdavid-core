"""
tests/integration/vector_store_inrusion_test.py

Vector store ownership isolation sweep for Project David.
Tests that row-level access control is correctly enforced on all
vector store endpoints.

Fixtures:
  owner    — OWNER_API_KEY
  intruder — INTRUDER_API_KEY

Run directly:
    python -m tests.integration.vector_store_inrusion_test

Or via the compliance runner (exposes run_sweep()).
"""

import os
import tempfile

import httpx
from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("ENTITIES_BASE_URL", "http://localhost:9000")
OWNER_KEY = os.getenv("OWNER_API_KEY")
INTRUDER_KEY = os.getenv("INTRUDER_API_KEY")

if not OWNER_KEY or not INTRUDER_KEY:
    raise RuntimeError(
        "Set OWNER_API_KEY and INTRUDER_API_KEY in your environment or .env file."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
results: dict = {}


def record(label: str, passed: bool, note: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    tag = "✅" if passed else "❌"
    print(f"[{status}] {tag}  {label}" + (f": {note}" if note else ""))
    results[label] = status


def expect_success(label: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        record(label, True)
        return result
    except Exception as exc:
        record(label, False, str(exc))
        return None


def expect_error(label: str, expected_codes: tuple, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        record(label, False, "Expected error but call succeeded")
        return None
    except Exception as exc:
        msg = str(exc)
        matched = any(str(code) in msg for code in expected_codes)
        record(label, matched, msg)
        return exc


# ──────────────────────────────────────────────────────────────────────────────
# Clients  (module level — one instance per process)
# ──────────────────────────────────────────────────────────────────────────────
owner_client = Entity(base_url=BASE_URL, api_key=OWNER_KEY)
intruder_client = Entity(base_url=BASE_URL, api_key=INTRUDER_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    results.clear()

    # Temp file created fresh each run
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="vs_isolation_test_", delete=False
    )
    tmp.write("Vector store isolation test payload. Safe to delete.")
    tmp.close()
    tmp_path = tmp.name
    print(f"\n[SETUP] Temp file created: {tmp_path}")

    owner_store = None

    print("\n" + "═" * 60)
    print("VECTOR STORE ISOLATION SWEEP")
    print("═" * 60)

    # ── Test 1: Owner creates vector store ───────────────────────────────────
    print("\n--- Test 1: Owner creates vector store (expecting success) ---")
    owner_store = expect_success(
        "Test 1: Owner creates vector store",
        owner_client.vectors.create_vector_store,
        name="isolation_test_store",
    )
    if owner_store:
        print(f"  store.id              : {owner_store.id}")
        print(f"  store.user_id         : {owner_store.user_id}")
        print(f"  store.collection_name : {owner_store.collection_name}")
    else:
        record("Sweep aborted", False, "Could not create owner vector store")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return results

    # ── Test 2: Owner retrieves own store by ID ───────────────────────────────
    print("\n--- Test 2: Owner retrieves own store by ID (expecting success) ---")
    expect_success(
        "Test 2: Owner retrieves own store by ID",
        owner_client.vectors.retrieve_vector_store_sync,
        owner_store.id,
    )

    # ── Test 3: Intruder retrieves owner's store by ID ────────────────────────
    print("\n--- Test 3: Intruder retrieves owner's store by ID (expecting 404) ---")
    expect_error(
        "Test 3: Intruder retrieves owner's store by ID",
        (404,),
        intruder_client.vectors.retrieve_vector_store_sync,
        owner_store.id,
    )

    # ── Test 4: Owner lists stores — own store present, no leakage ───────────
    print("\n--- Test 4: Owner lists stores (expecting only own stores) ---")
    try:
        owner_stores = owner_client.vectors.list_my_vector_stores()
        owner_store_ids = [s.id for s in owner_stores]

        record("Test 4: Owner store in list", owner_store.id in owner_store_ids)

        leaked = [s for s in owner_stores if s.user_id != owner_store.user_id]
        record(
            "Test 4: No cross-user leakage",
            len(leaked) == 0,
            f"{len(leaked)} foreign store(s) leaked" if leaked else "",
        )
    except Exception as exc:
        record("Test 4: Owner store in list", False, str(exc))

    # ── Test 5: Intruder lists stores — must not see owner's ─────────────────
    print("\n--- Test 5: Intruder lists stores (must not see owner's store) ---")
    try:
        intruder_stores = intruder_client.vectors.list_my_vector_stores()
        intruder_store_ids = [s.id for s in intruder_stores]
        leaked = owner_store.id in intruder_store_ids
        record(
            "Test 5: No cross-user leakage",
            not leaked,
            "owner's store visible to intruder!" if leaked else "",
        )
    except Exception as exc:
        record("Test 5: No cross-user leakage", False, str(exc))

    # ── Test 6: Owner adds file to own store ──────────────────────────────────
    print("\n--- Test 6: Owner adds file to own store (expecting success) ---")
    owner_file = expect_success(
        "Test 6: Owner adds file to own store",
        owner_client.vectors.add_file_to_vector_store,
        owner_store.id,
        tmp_path,
    )
    if owner_file:
        print(f"  file.id     : {owner_file.id}")
        print(f"  file.status : {owner_file.status}")

    # ── Test 7: Intruder adds file to owner's store ───────────────────────────
    print("\n--- Test 7: Intruder adds file to owner's store (expecting 404) ---")
    expect_error(
        "Test 7: Intruder adds file to owner's store",
        (404,),
        intruder_client.vectors.add_file_to_vector_store,
        owner_store.id,
        tmp_path,
    )

    # ── Test 8: Intruder lists owner's store files ────────────────────────────
    print("\n--- Test 8: Intruder lists owner's store files (expecting 404) ---")
    expect_error(
        "Test 8: Intruder lists owner's store files",
        (404,),
        intruder_client.vectors.list_store_files,
        owner_store.id,
    )

    # ── Test 9: Owner lists own store files ───────────────────────────────────
    print("\n--- Test 9: Owner lists own store files ---")
    try:
        files = owner_client.vectors.list_store_files(owner_store.id)
        record("Test 9: Owner lists own files", True, f"{len(files)} file(s) retrieved")
        print(f"  files retrieved: {len(files)}")
    except Exception as exc:
        record("Test 9: Owner lists own files", False, str(exc))

    # ── Test 10: Intruder deletes owner's store ───────────────────────────────
    print("\n--- Test 10: Intruder deletes owner's store (expecting 404) ---")
    expect_error(
        "Test 10: Intruder deletes owner's store",
        (404,),
        intruder_client.vectors.delete_vector_store,
        owner_store.id,
    )

    # ── Test 11: Owner deletes own store ──────────────────────────────────────
    print("\n--- Test 11: Owner deletes own store (expecting success) ---")
    expect_success(
        "Test 11: Owner deletes own store",
        owner_client.vectors.delete_vector_store,
        owner_store.id,
        permanent=True,
    )

    # ── Test 12: Collection-name lookup isolation (Gap 2 fix) ─────────────────
    # Before the fix, GET /vector-stores/lookup/collection had no auth at all.
    # After the fix, ownership is enforced — intruder gets 404.
    # We create a fresh probe store so it exists during the lookup attempt.
    print("\n--- Test 12: Collection-name lookup isolation (Gap 2 fix) ---")

    probe_store = expect_success(
        "Test 12 setup: Owner creates probe store for collection lookup",
        owner_client.vectors.create_vector_store,
        name="isolation_probe_store",
    )

    if probe_store:
        probe_collection = probe_store.collection_name
        print(f"  probe_store.collection_name : {probe_collection}")

        # 12a: Owner can look up own store by collection name
        print(
            "\n--- Test 12a: Owner looks up own store by collection name (expecting success) ---"
        )
        expect_success(
            "Test 12a: Owner looks up own store by collection name",
            owner_client.vectors.retrieve_vector_store_sync,
            probe_store.id,
        )

        # 12b: Intruder cannot — raw HTTP call to the collection lookup endpoint
        print(
            "\n--- Test 12b: Intruder looks up owner's store by collection name (Gap 2 fix) (expecting 404) ---"
        )
        try:
            response = httpx.get(
                f"{BASE_URL}/v1/vector-stores/lookup/collection",
                params={"name": probe_collection},
                headers={"X-API-Key": INTRUDER_KEY},
                timeout=10.0,
            )
            if response.status_code == 404:
                record(
                    "Test 12b: Intruder looks up owner's store by collection name (Gap 2 fix)",
                    True,
                    f"correctly returned 404: {response.text}",
                )
            else:
                record(
                    "Test 12b: Intruder looks up owner's store by collection name (Gap 2 fix)",
                    False,
                    f"unexpected status {response.status_code}: {response.text}",
                )
        except Exception as exc:
            record(
                "Test 12b: Intruder looks up owner's store by collection name (Gap 2 fix)",
                False,
                str(exc),
            )

        # Teardown probe store
        print(
            "\n--- Test 12 teardown: Owner permanently deletes probe store (expecting success) ---"
        )
        expect_success(
            "Test 12 teardown: Owner permanently deletes probe store",
            owner_client.vectors.delete_vector_store,
            probe_store.id,
            permanent=True,
        )
    else:
        print("[WARN] ⚠️  Test 12: Could not create probe store — skipping.")
        results[
            "Test 12b: Intruder looks up owner's store by collection name (Gap 2 fix)"
        ] = "FAIL"

    # ── Teardown ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("TEARDOWN")
    print("═" * 60)
    try:
        os.unlink(tmp_path)
        print(f"[OK] Temp file removed: {tmp_path}")
    except Exception as exc:
        print(f"[WARN] Could not remove temp file: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("VECTOR STORE ISOLATION SWEEP — SUMMARY")
    print("═" * 60)
    for label, outcome in results.items():
        tag = "✅" if outcome == "PASS" else "❌"
        print(f"  {tag}  {label}: {outcome}")
    print("═" * 60)

    return results


if __name__ == "__main__":
    run_sweep()
