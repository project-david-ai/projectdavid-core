"""Regression tests for the static public API publication policy."""

import copy
import json

import pytest

from scripts.generate_public_openapi import (
    PUBLIC_GROUPS,
    build_public_openapi,
    operation_inventory,
)


@pytest.fixture
def source():
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Project David API", "version": "1.0.0"},
        "paths": {},
        "components": {
            "schemas": {
                "Parent": {
                    "type": "object",
                    "properties": {"nested": {"$ref": "#/components/schemas/Child"}},
                },
                "Child": {"type": "string"},
                "Internal": {"type": "string"},
            },
            "securitySchemes": {
                "APIKeyHeader": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                }
            },
        },
    }
    index = 0
    for operations in PUBLIC_GROUPS.values():
        for entry in operations:
            method, path = entry.split(" ", 1)
            index += 1
            operation = {
                "operationId": f"op{index}",
                "tags": ["Duplicate", "Duplicate"],
                "responses": {"200": {"description": "OK"}},
                "security": [{"APIKeyHeader": []}],
            }
            document["paths"].setdefault(path, {})[method.lower()] = operation

    document["paths"]["/v1/health"]["get"]["responses"]["200"]["content"] = {
        "application/json": {"schema": {"$ref": "#/components/schemas/Parent"}}
    }
    document["paths"]["/v1/vector-stores"]["post"]["parameters"] = [
        {"name": "owner_id", "in": "query", "schema": {"type": "string"}}
    ]
    document["paths"]["/v1/admin/example"] = {
        "get": {"operationId": "internal", "responses": {"200": {"description": "OK"}}}
    }
    document["paths"]["/v1/actions/{action_id}"]["delete"] = {
        "operationId": "internal_delete",
        "responses": {"204": {"description": "OK"}},
    }
    return document


def test_exact_allowlist_and_transitive_component_closure(source):
    public = build_public_openapi(source)
    expected = {entry for group in PUBLIC_GROUPS.values() for entry in group}
    assert set(operation_inventory(public)) == expected
    assert "/v1/admin/example" not in public["paths"]
    assert "delete" not in public["paths"]["/v1/actions/{action_id}"]
    assert set(public["components"]["schemas"]) == {"Parent", "Child"}
    assert "APIKeyHeader" in public["components"]["securitySchemes"]
    assert public["paths"]["/v1/health"]["get"]["tags"] == ["Health"]


def test_missing_approved_operation_fails(source):
    del source["paths"]["/v1/runs"]["post"]
    with pytest.raises(ValueError, match="Missing approved operations"):
        build_public_openapi(source)


def test_unknown_operation_does_not_automatically_publish(source):
    source["paths"]["/v1/future"] = {"get": {"operationId": "new", "tags": ["Runs"]}}
    assert "/v1/future" not in build_public_openapi(source)["paths"]


def test_broken_transitive_reference_fails(source):
    del source["components"]["schemas"]["Child"]
    with pytest.raises(ValueError, match="Unresolved reference"):
        build_public_openapi(source)


def test_external_reference_fails(source):
    source["components"]["schemas"]["Child"] = {
        "$ref": "https://example.com/schema.json"
    }
    with pytest.raises(ValueError, match="Unsupported reference"):
        build_public_openapi(source)


def test_authentication_drift_fails(source):
    del source["paths"]["/v1/runs"]["post"]["security"]
    with pytest.raises(ValueError, match="Authentication contract changed"):
        build_public_openapi(source)


def test_duplicate_operation_ids_fail(source):
    source["paths"]["/v1/runs"]["post"]["operationId"] = source["paths"]["/v1/runs"][
        "get"
    ]["operationId"]
    with pytest.raises(ValueError, match="duplicate operationId"):
        build_public_openapi(source)


def test_deterministic_and_does_not_mutate_source(source):
    original = copy.deepcopy(source)
    assert json.dumps(build_public_openapi(source)) == json.dumps(
        build_public_openapi(source)
    )
    assert source == original


def test_only_explicit_server_is_published(source):
    source["servers"] = [{"url": "http://internal.example"}]
    assert "servers" not in build_public_openapi(source)
    public = build_public_openapi(source, server="https://api.example.com/")
    assert public["servers"] == [{"url": "https://api.example.com"}]
    with pytest.raises(ValueError, match="explicit HTTP"):
        build_public_openapi(source, server="file:///tmp/api")


def test_public_vector_store_creation_omits_admin_delegation(source):
    public = build_public_openapi(source)
    assert "parameters" not in public["paths"]["/v1/vector-stores"]["post"]
    assert source["paths"]["/v1/vector-stores"]["post"]["parameters"]
