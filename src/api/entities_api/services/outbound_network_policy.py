"""Shared outbound address validation and operator-controlled MCP policy."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
from urllib.parse import urlsplit


class OutboundPolicyError(RuntimeError):
    """A destination could not be safely authorized."""


def public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if (
        address == ipaddress.ip_address("168.63.129.16")
        or not address.is_global
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # Exclude transition mechanisms and scoped addresses. Restrict IPv6 to
        # ordinary global unicast; IPv4-embedded destinations are not needed.
        if (
            address.scope_id
            or address.ipv4_mapped
            or address.sixtofour
            or address.teredo
        ):
            return False
        if (
            address not in ipaddress.IPv6Network("2000::/3")
            or address in ipaddress.IPv6Network("2001::/23")
            or address in ipaddress.IPv6Network("3fff::/20")
        ):
            return False
    elif address in ipaddress.IPv4Network(
        "192.0.0.0/24"
    ) or address in ipaddress.IPv4Network("192.88.99.0/24"):
        return False
    return True


async def resolve_addresses(host: str, port: int) -> list[str]:
    records = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return list(dict.fromkeys(record[4][0] for record in records))


def origin(value: str) -> tuple[str, str, int]:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(ord(c) <= 32 or ord(c) == 127 or c == "\\" for c in value)
            or "%" in parsed.hostname
        ):
            raise ValueError
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        return (
            parsed.scheme,
            parsed.hostname.lower(),
            port or (443 if parsed.scheme == "https" else 80),
        )
    except ValueError:
        raise OutboundPolicyError("Invalid outbound endpoint") from None


def _operator_networks(
    target: tuple[str, str, int],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Exact origin -> explicitly permitted CIDRs; never a tenant setting."""
    try:
        raw = json.loads(os.environ.get("PROJECT_DAVID_MCP_PRIVATE_ORIGINS", "{}"))
        if not isinstance(raw, dict):
            raise ValueError
        matches: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()
        for key, values in raw.items():
            parsed = urlsplit(key)
            if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
                raise ValueError
            configured = origin(key)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(v, str) for v in values)
            ):
                raise ValueError
            networks = tuple(ipaddress.ip_network(v, strict=True) for v in values)
            if any(n.prefixlen == 0 for n in networks):
                raise ValueError
            if configured == target:
                matches = networks
        return matches
    except (ValueError, TypeError, OutboundPolicyError):
        raise OutboundPolicyError("Invalid operator MCP network policy") from None


async def resolve_mcp_destination(target: tuple[str, str, int]) -> str:
    scheme, host, port = target
    networks = _operator_networks(target)
    if scheme != "https" and not networks:
        raise OutboundPolicyError("Public MCP endpoints must use HTTPS")
    try:
        addresses = await resolve_addresses(host, port)
    except OSError:
        raise OutboundPolicyError("MCP destination resolution failed") from None

    def permitted(value: str) -> bool:
        if not networks:
            return public_address(value)
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        if address.is_multicast or address.is_unspecified or address.is_link_local:
            return False
        if address == ipaddress.ip_address("168.63.129.16"):
            return False
        if isinstance(address, ipaddress.IPv6Address) and (
            address.scope_id
            or address.ipv4_mapped
            or address.sixtofour
            or address.teredo
        ):
            return False
        return any(address in network for network in networks)

    if not addresses or not all(permitted(address) for address in addresses):
        raise OutboundPolicyError("MCP destination is not permitted")
    return next((address for address in addresses if ":" not in address), addresses[0])
