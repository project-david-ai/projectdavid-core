import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.api.entities_api.services import outbound_network_policy as policy


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    monkeypatch.delenv("PROJECT_DAVID_MCP_PRIVATE_ORIGINS", raising=False)


@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["168.63.129.16"],
        ["8.8.8.8", "::1"],
    ],
)
def test_public_policy_rejects_unsafe_dns(monkeypatch, addresses):
    monkeypatch.setattr(policy, "resolve_addresses", AsyncMock(return_value=addresses))
    with pytest.raises(policy.OutboundPolicyError):
        asyncio.run(policy.resolve_mcp_destination(("https", "example.com", 443)))


def test_public_https(monkeypatch):
    monkeypatch.setattr(
        policy, "resolve_addresses", AsyncMock(return_value=["8.8.8.8"])
    )
    assert (
        asyncio.run(policy.resolve_mcp_destination(("https", "example.com", 443)))
        == "8.8.8.8"
    )


def test_public_http_blocked_before_dns(monkeypatch):
    lookup = AsyncMock()
    monkeypatch.setattr(policy, "resolve_addresses", lookup)
    with pytest.raises(policy.OutboundPolicyError):
        asyncio.run(policy.resolve_mcp_destination(("http", "example.com", 80)))
    lookup.assert_not_awaited()


def test_exact_operator_origin_and_network(monkeypatch):
    monkeypatch.setenv(
        "PROJECT_DAVID_MCP_PRIVATE_ORIGINS",
        json.dumps({"http://internal:8080": ["10.0.0.0/24"]}),
    )
    lookup = AsyncMock(return_value=["10.0.0.2"])
    monkeypatch.setattr(policy, "resolve_addresses", lookup)
    assert (
        asyncio.run(policy.resolve_mcp_destination(("http", "internal", 8080)))
        == "10.0.0.2"
    )
    with pytest.raises(policy.OutboundPolicyError):
        asyncio.run(policy.resolve_mcp_destination(("http", "internal", 8081)))
    lookup.return_value = ["10.0.1.2"]
    with pytest.raises(policy.OutboundPolicyError):
        asyncio.run(policy.resolve_mcp_destination(("http", "internal", 8080)))


def test_malformed_policy_fails_closed(monkeypatch):
    monkeypatch.setenv("PROJECT_DAVID_MCP_PRIVATE_ORIGINS", "not-json")
    with pytest.raises(policy.OutboundPolicyError):
        asyncio.run(policy.resolve_mcp_destination(("https", "example.com", 443)))
