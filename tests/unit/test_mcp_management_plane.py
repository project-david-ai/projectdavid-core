"""MCP management-plane routing contracts."""

from fastapi.routing import APIRoute

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.routers import api_router
from src.api.entities_api.routers.mcp_router import router as mcp_router

EXPECTED = {
    (
        "/mcp/servers",
        "POST",
    ),
    (
        "/mcp/servers",
        "GET",
    ),
    (
        "/mcp/servers/{server_id}",
        "GET",
    ),
    (
        "/mcp/servers/{server_id}",
        "PATCH",
    ),
    (
        "/mcp/servers/{server_id}",
        "DELETE",
    ),
    (
        "/mcp/servers/{server_id}/tools",
        "GET",
    ),
    (
        "/assistants/{assistant_id}/mcp-tools",
        "GET",
    ),
    (
        "/assistants/{assistant_id}/mcp-tools",
        "POST",
    ),
    (
        "/assistants/{assistant_id}/mcp-tools",
        "DELETE",
    ),
}


def route_contracts(router):
    contracts = set()

    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue

        for method in route.methods:
            if method in {
                "HEAD",
                "OPTIONS",
            }:
                continue

            contracts.add(
                (
                    route.path,
                    method,
                )
            )

    return contracts


def test_mcp_management_routes_are_exactly_the_expected_surface():
    assert route_contracts(mcp_router) == EXPECTED


def test_mcp_management_routes_are_mounted():
    mounted = route_contracts(api_router)

    assert EXPECTED <= mounted


def test_every_mcp_endpoint_uses_house_auth_and_db_dependencies():
    routes = [route for route in mcp_router.routes if isinstance(route, APIRoute)]

    assert routes

    for route in routes:
        dependencies = {
            dependency.call for dependency in (route.dependant.dependencies)
        }

        assert get_api_key in dependencies
        assert get_db in dependencies
