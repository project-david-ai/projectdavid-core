"""Local Core tenant-isolation regression; no model inference is required.

Run explicitly with python -m tests.integration.tenant_ownership_test.
Use two distinct disposable test accounts, not production credentials.
"""

import os
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from projectdavid import Entity
from projectdavid_common import UtilsInterface, ValidationInterface


def request(client, method, path, expected, **kwargs):
    response = client.request(method, path, **kwargs)
    if response.status_code not in expected:
        raise AssertionError(
            f"{method} {path}: expected {expected}, got {response.status_code}"
        )
    return response.json() if response.is_success and response.content else None


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"PASS: {label}")


def run_sweep():
    load_dotenv()
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("This destructive-fixture sweep requires a local Core URL.")
    owner_key = os.environ["OWNER_API_KEY"]
    intruder_key = os.environ["INTRUDER_API_KEY"]
    if not owner_key or not intruder_key or owner_key == intruder_key:
        raise RuntimeError("Two distinct test credentials are required.")

    with (
        httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": owner_key},
            timeout=30.0,
            trust_env=False,
        ) as owner,
        httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": intruder_key},
            timeout=30.0,
            trust_env=False,
        ) as intruder,
    ):
        owner_identity = request(owner, "GET", "/v1/auth/validate-key", (200,))
        intruder_identity = request(intruder, "GET", "/v1/auth/validate-key", (200,))
        require(
            owner_identity["valid"] is True
            and intruder_identity["valid"] is True
            and owner_identity["user_id"] != intruder_identity["user_id"],
            "Both credentials authenticate as distinct users",
        )
        sdk = Entity(base_url=base_url, api_key=owner_key)
        assistant = thread = run = None
        action_ids = []
        cleanup_errors = []
        try:
            assistant = sdk.assistants.create_assistant(
                name="Tenant isolation regression", model="gpt-oss-120b"
            )
            thread = sdk.threads.create_thread()
            run = sdk.runs.create_run(assistant_id=assistant.id, thread_id=thread.id)
            require(run.user_id == owner_identity["user_id"], "Owner creates run")
            action_payload = ValidationInterface.ActionCreate(
                id=UtilsInterface.IdentifierService.generate_action_id(),
                run_id=run.id,
                tool_name="tenant_isolation_probe",
                tool_call_id="call_tenant_isolation_probe",
                function_args={"probe": "synthetic"},
                status="pending",
            ).model_dump(mode="json")
            action = request(owner, "POST", "/v1/actions", (200,), json=action_payload)
            action_ids.append(action["id"])
            require(action["run_id"] == run.id, "Owner creates action")
            action_path = f"/v1/actions/{action['id']}"
            pending_path = f"/v1/actions/pending/{run.id}"
            status_path = f"/v1/runs/{run.id}/actions/status"
            fields_path = f"/v1/runs/{run.id}/fields"
            pending_before = request(owner, "GET", pending_path, (200,))
            require(
                any(item["action_id"] == action["id"] for item in pending_before),
                "Owner sees pending action",
            )
            require(
                any(
                    item["id"] == action["id"]
                    for item in request(owner, "GET", status_path, (200,))
                ),
                "Owner lists actions by status",
            )
            original = request(owner, "GET", action_path, (200,))
            require(original["id"] == action["id"], "Owner retrieves action")

            foreign_payload = dict(action_payload)
            foreign_payload["id"] = (
                UtilsInterface.IdentifierService.generate_action_id()
            )
            update_payload = {"status": "completed", "result": {"probe": "foreign"}}
            for method, path, kwargs in (
                ("POST", "/v1/actions", {"json": foreign_payload}),
                ("GET", action_path, {}),
                ("PUT", action_path, {"json": update_payload}),
                ("GET", pending_path, {}),
                ("GET", status_path, {}),
                ("DELETE", action_path, {}),
            ):
                request(intruder, method, path, (403, 404), **kwargs)
                print(f"PASS: Foreign {method} {path} denied")
            require(
                request(owner, "GET", action_path, (200,)) == original,
                "Foreign mutations leave original action unchanged",
            )
            require(
                request(owner, "GET", pending_path, (200,)) == pending_before,
                "Foreign creation leaves pending actions unchanged",
            )

            for payload in ({}, {"not_a_mutable_field": True}):
                request(intruder, "PATCH", fields_path, (403, 404), json=payload)
                request(owner, "PATCH", fields_path, (200,), json=payload)
                print("PASS: No-op run update enforces ownership")

            messages_path = f"/v1/threads/{thread.id}/messages"
            before = request(owner, "GET", messages_path, (200,))
            message_payload = {
                "thread_id": thread.id,
                "assistant_id": assistant.id,
                "sender_id": assistant.id,
                "role": "assistant",
                "content": "Synthetic isolation probe",
            }
            request(
                intruder,
                "POST",
                "/v1/messages/assistant",
                (403, 404),
                json=message_payload,
            )
            require(
                request(owner, "GET", messages_path, (200,)) == before,
                "Foreign assistant write leaves thread unchanged",
            )
            saved = request(
                owner, "POST", "/v1/messages/assistant", (200,), json=message_payload
            )
            require(
                saved["thread_id"] == thread.id
                and saved["content"] == message_payload["content"],
                "Owner assistant persistence remains compatible",
            )

            processing = request(
                owner, "PUT", action_path, (200,), json={"status": "processing"}
            )
            require(
                processing["status"] == "processing", "Owner claims external action"
            )
            tool_payload = {
                "thread_id": thread.id,
                "assistant_id": assistant.id,
                "role": "tool",
                "content": "Synthetic tool result",
                "tool_call_id": action_payload["tool_call_id"],
                "tool_id": action["id"],
            }
            request(
                intruder, "POST", "/v1/messages/tools", (403, 404), json=tool_payload
            )
            tool_message = request(
                owner, "POST", "/v1/messages/tools", (200,), json=tool_payload
            )
            require(
                tool_message["thread_id"] == thread.id
                and tool_message["role"] == "tool",
                "Owner submits tool result",
            )
            result = {"probe": "owner"}
            updated = request(
                owner,
                "PUT",
                action_path,
                (200,),
                json={"status": "completed", "result": result},
            )
            require(
                updated["status"] == "completed" and updated["result"] == result,
                "Owner completes external action",
            )
            request(owner, "DELETE", action_path, (204,))
            action_ids.remove(action["id"])
            request(owner, "GET", action_path, (404,))
            print("PASS: Owner deletes test action")
        finally:
            # Cleanup only resources created by this sweep. A failed cleanup
            # must never turn an unsuccessful test into a PASS.
            cleanup = [
                ("DELETE", f"/v1/actions/{aid}", (204, 404)) for aid in action_ids
            ]
            if run is not None:
                cleanup.append(("POST", f"/v1/runs/{run.id}/cancel", (200,)))
            if thread is not None:
                cleanup.append(("DELETE", f"/v1/threads/{thread.id}", (200, 204)))
            if assistant is not None:
                cleanup.append(("DELETE", f"/v1/assistants/{assistant.id}", (204,)))
            for method, path, expected in cleanup:
                try:
                    request(owner, method, path, expected)
                except Exception as exc:
                    cleanup_errors.append(f"{method} {path}: {type(exc).__name__}")
            for error in cleanup_errors:
                print(f"CLEANUP FAILED: {error}")
        if cleanup_errors:
            raise RuntimeError("Test fixtures require cleanup; see paths above.")
    print("PASS: Core tenant ownership regression")


if __name__ == "__main__":
    run_sweep()
