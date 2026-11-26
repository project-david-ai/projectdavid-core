"""Generate the hosted developer OpenAPI contract from a Core schema snapshot.

Run from the Core repository root:
    python scripts/generate_public_openapi.py INPUT OUTPUT [--server URL]

No runtime imports or network access. New routes are excluded until explicitly
approved. A missing approved route fails generation instead of silently drifting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

HTTP_METHODS = frozenset(
    ("get", "post", "put", "patch", "delete", "head", "options", "trace")
)

# Explicit method + path policy; never publish by tag or path prefix alone.
PUBLIC_GROUPS = {
    "Health": ("GET /v1/health",),
    "Inference": ("POST /v1/completions",),
    "Assistants": (
        "GET /v1/assistants",
        "POST /v1/assistants",
        "GET /v1/assistants/summaries",
        "GET /v1/assistants/{assistant_id}",
        "PUT /v1/assistants/{assistant_id}",
        "DELETE /v1/assistants/{assistant_id}",
    ),
    "Threads": (
        "POST /v1/threads",
        "GET /v1/threads/{thread_id}",
        "PUT /v1/threads/{thread_id}",
        "DELETE /v1/threads/{thread_id}",
        "GET /v1/threads/user/{user_id}",
        "GET /v1/threads/user/{user_id}/records",
        "PUT /v1/threads/{thread_id}/metadata",
    ),
    "Messages": (
        "POST /v1/messages",
        "GET /v1/messages/{message_id}",
        "DELETE /v1/messages/{message_id}",
        "GET /v1/threads/{thread_id}/messages",
        "POST /v1/messages/tools",
    ),
    "Runs": (
        "POST /v1/runs",
        "GET /v1/runs",
        "GET /v1/runs/{run_id}",
        "PUT /v1/runs/{run_id}/metadata",
        "POST /v1/runs/{run_id}/cancel",
        "GET /v1/runs/{run_id}/events",
        "GET /v1/assistants/{assistant_id}/runs",
        "GET /v1/threads/{thread_id}/runs",
    ),
    "Execution Accounting": (
        "GET /v1/runs/{run_id}/execution",
        "GET /v1/runs/{run_id}/delegations",
        "GET /v1/runs/{run_id}/execution-events",
    ),
    "Actions": (
        "GET /v1/actions/{action_id}",
        "PUT /v1/actions/{action_id}",
        "GET /v1/actions/pending/{run_id}",
        "GET /v1/runs/{run_id}/actions/status",
    ),
    "Scratchpads": (
        "GET /v1/scratchpads",
        "POST /v1/scratchpads",
        "GET /v1/scratchpads/{scratchpad_id}",
        "PATCH /v1/scratchpads/{scratchpad_id}",
        "DELETE /v1/scratchpads/{scratchpad_id}",
        "GET /v1/threads/{thread_id}/scratchpad",
        "GET /v1/scratchpads/{scratchpad_id}/content",
        "PUT /v1/scratchpads/{scratchpad_id}/content",
        "DELETE /v1/scratchpads/{scratchpad_id}/content",
        "POST /v1/scratchpads/{scratchpad_id}/entries",
        "GET /v1/scratchpads/{scratchpad_id}/entries",
        "DELETE /v1/scratchpads/{scratchpad_id}/entries",
        "POST /v1/scratchpads/{scratchpad_id}/clear",
    ),
    "Files": (
        "POST /v1/uploads",
        "GET /v1/files",
        "GET /v1/files/{file_id}",
        "DELETE /v1/files/{file_id}",
        "PATCH /v1/files/{file_id}/soft-delete",
        "GET /v1/files/{file_id}/signed-url",
        "GET /v1/files/{file_id}/base64",
    ),
    "Vector Stores": (
        "POST /v1/vector-stores",
        "GET /v1/vector-stores",
        "GET /v1/vector-stores/lookup/collection",
        "GET /v1/vector-stores/{vector_store_id}",
        "DELETE /v1/vector-stores/{vector_store_id}",
        "POST /v1/vector-stores/{vector_store_id}/files",
        "GET /v1/vector-stores/{vector_store_id}/files",
        "DELETE /v1/vector-stores/{vector_store_id}/files",
        "PATCH /v1/vector-stores/{vector_store_id}/files/{file_id}",
        "POST /v1/vector-stores/{vector_store_id}/vectors",
        "POST /v1/vector-stores/{vector_store_id}/search",
    ),
    "MCP": (
        "GET /v1/mcp/servers",
        "POST /v1/mcp/servers",
        "GET /v1/mcp/servers/{server_id}",
        "PATCH /v1/mcp/servers/{server_id}",
        "DELETE /v1/mcp/servers/{server_id}",
        "PUT /v1/mcp/servers/{server_id}/credential",
        "GET /v1/mcp/servers/{server_id}/tools",
        "GET /v1/assistants/{assistant_id}/mcp-tools",
        "POST /v1/assistants/{assistant_id}/mcp-tools",
        "DELETE /v1/assistants/{assistant_id}/mcp-tools",
    ),
    "Provider Keys": (
        "GET /v1/provider-keys",
        "POST /v1/provider-keys",
        "GET /v1/provider-keys/{provider_key_id}",
        "PATCH /v1/provider-keys/{provider_key_id}",
        "DELETE /v1/provider-keys/{provider_key_id}",
    ),
    "API Keys": (
        "POST /v1/users/{user_id}/apikeys",
        "GET /v1/users/{user_id}/apikeys",
        "GET /v1/users/{user_id}/apikeys/{key_prefix}",
        "DELETE /v1/users/{user_id}/apikeys/{key_prefix}",
    ),
}


def operation_inventory(document):
    return {
        f"{method.upper()} {path}": operation
        for path, item in document["paths"].items()
        for method, operation in item.items()
        if method in HTTP_METHODS
    }


def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def pointer(document, ref):
    """Resolve a local JSON Pointer; reject unresolved or external references."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError(f"Unsupported reference: {ref!r}")
    node = document
    try:
        for token in ref[2:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            node = node[int(token)] if isinstance(node, list) else node[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Unresolved reference: {ref}") from exc
    return node


def component_dependencies(node):
    for key, value in walk(node):
        if key == "$ref":
            yield value
        elif key == "security" and isinstance(value, list):
            for requirement in value:
                for name in requirement:
                    escaped = name.replace("~", "~0").replace("/", "~1")
                    yield f"#/components/securitySchemes/{escaped}"
        elif key in ("$dynamicRef", "operationRef"):
            raise ValueError(f"Unsupported {key}; review publication policy")
        elif key in ("callbacks", "links", "discriminator"):
            # The current FastAPI contract has none. Explicit review is required
            # before supporting implicit links to other operations or schemas.
            raise ValueError(f"Unsupported {key}; review publication policy")


def retain_components(source, public):
    retained = {}
    pending = list(component_dependencies(public))
    seen = set()
    while pending:
        ref = pending.pop()
        if ref in seen:
            continue
        pointer(source, ref)
        parts = ref.split("/")
        if len(parts) < 4 or parts[1] != "components":
            raise ValueError(f"Reference outside components: {ref}")
        section, name = (
            token.replace("~1", "/").replace("~0", "~") for token in parts[2:4]
        )
        component = source["components"][section][name]
        retained.setdefault(section, {})[name] = copy.deepcopy(component)
        seen.add(ref)
        pending.extend(component_dependencies(component))
    return {
        section: {name: entries[name] for name in sorted(entries)}
        for section, entries in sorted(retained.items())
    }


def build_public_openapi(source, *, server=None):
    if source.get("openapi") != "3.1.0":
        raise ValueError("Expected Core OpenAPI 3.1.0; review generator on upgrade")
    if not source.get("info", {}).get("version"):
        raise ValueError("Missing Core API version")
    if source.get("info", {}).get("title") != "Project David API":
        raise ValueError("Unexpected Core API title")
    if source.get("webhooks") or source.get("security"):
        raise ValueError("Review new root-level webhooks/security before publication")

    approved = {
        operation: group
        for group, operations in PUBLIC_GROUPS.items()
        for operation in operations
    }
    if len(approved) != sum(map(len, PUBLIC_GROUPS.values())):
        raise ValueError("Duplicate operation in publication policy")
    inventory = operation_inventory(source)
    missing = sorted(approved.keys() - inventory.keys())
    if missing:
        raise ValueError("Missing approved operations:\n" + "\n".join(missing))

    public = {
        "openapi": source["openapi"],
        "info": copy.deepcopy(source["info"]),
        "tags": [{"name": group} for group in PUBLIC_GROUPS],
        "paths": {},
    }
    if server is not None:
        parsed = urlsplit(server)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Server must be an explicit HTTP(S) API base URL")
        public["servers"] = [{"url": server.rstrip("/")}]

    for operation, group in approved.items():
        method, path = operation.split(" ", 1)
        item = source["paths"][path]
        if set(item) - HTTP_METHODS:
            raise ValueError(f"New path-level metadata requires review: {path}")
        selected = copy.deepcopy(inventory[operation])
        if "servers" in selected:
            raise ValueError(f"Operation-level server requires review: {operation}")
        selected["tags"] = [group]
        # Publish the ordinary customer contract, not administrative delegation.
        if operation == "POST /v1/vector-stores":
            selected["parameters"] = [
                p
                for p in selected.get("parameters", [])
                if not (p.get("in") == "query" and p.get("name") == "owner_id")
            ]
            if not selected["parameters"]:
                selected.pop("parameters")
            selected["description"] = (
                "Creates a vector store for the authenticated user."
            )
        if group == "API Keys":
            selected["description"] = (
                selected.get("description", "")
                + "\n\nUse your own user ID for self-service key management."
            ).strip()
        public["paths"].setdefault(path, {})[method.lower()] = selected

    public["components"] = retain_components(source, public)
    # Omit the admin-labelled illustrative example, not any request field.
    key_schema = public["components"].get("schemas", {}).get("ApiKeyCreateRequest")
    if key_schema and "examples" in key_schema:
        key_schema["examples"] = [
            example
            for example in key_schema["examples"]
            if example.get("key_name") != "Permanent Admin Key"
        ]

    validate_public_contract(public, approved)
    return public


def validate_public_contract(public, approved):
    inventory = operation_inventory(public)
    if set(inventory) != set(approved):
        raise ValueError("Public operation inventory differs from allowlist")
    ids = set()
    for name, operation in inventory.items():
        operation_id = operation.get("operationId")
        if not operation_id or operation_id in ids:
            raise ValueError(f"Missing or duplicate operationId: {name}")
        ids.add(operation_id)
        if operation.get("tags") != [approved[name]]:
            raise ValueError(f"Invalid public tag: {name}")
        if name not in ("GET /v1/health", "POST /v1/completions"):
            if operation.get("security") != [{"APIKeyHeader": []}]:
                raise ValueError(f"Authentication contract changed: {name}")
        if not operation.get("responses"):
            raise ValueError(f"Missing responses: {name}")
    scheme = public["components"].get("securitySchemes", {}).get("APIKeyHeader")
    if scheme != {"type": "apiKey", "in": "header", "name": "X-API-Key"}:
        raise ValueError("Core API key header contract changed")
    for ref in component_dependencies(public):
        pointer(public, ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    public = build_public_openapi(source, server=args.server)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (
            json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    original = operation_inventory(source)
    selected = operation_inventory(public)
    print(f"Published {len(selected)} of {len(original)} operations")
    print(f"Retained {len(public['components'].get('schemas', {}))} component schemas")
    print(f"Output: {args.output}")
    if args.server is None:
        print("No API server configured; reference preview only.")
    print("Excluded operations:")
    for operation in sorted(original.keys() - selected.keys()):
        print(f"  {operation}")


if __name__ == "__main__":
    main()
