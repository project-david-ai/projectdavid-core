"""
tests/test_assistant_cache_mixin.py
────────────────────────────────────
Unit tests for AssistantCacheMixin four-level resolution chain.

Tests each resolution path in isolation and verifies the hard
ValueError on misconfiguration. No DB, Redis, or network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.entities_api.orchestration.mixins.assistant_cache_mixin import (
    AssistantCacheMixin,
)

# ---------------------------------------------------------------------------
# Helpers — minimal concrete classes for each path
# ---------------------------------------------------------------------------


class BareClass(AssistantCacheMixin):
    """No injection, no service registry, no redis — triggers path 4."""

    pass


class InjectedClass(AssistantCacheMixin):
    """Simulates path 1: cache injected directly."""

    def __init__(self, cache):
        self._assistant_cache = cache


class ServiceRegistryClass(AssistantCacheMixin):
    """Simulates path 2: _get_service() available on self."""

    def __init__(self, service_return):
        self._service_return = service_return

    def _get_service(self, cls):
        return self._service_return


class RedisClass(AssistantCacheMixin):
    """Simulates path 3: self.redis available, no injection, no registry."""

    def __init__(self, redis):
        self.redis = redis


# ---------------------------------------------------------------------------
# Path 1 — injected via __init__
# ---------------------------------------------------------------------------


def test_path1_injected_cache_returned():
    mock_cache = MagicMock(name="AssistantCache")
    obj = InjectedClass(cache=mock_cache)
    assert obj.assistant_cache is mock_cache


def test_path1_get_assistant_cache_delegates_to_property():
    mock_cache = MagicMock(name="AssistantCache")
    obj = InjectedClass(cache=mock_cache)
    assert obj.get_assistant_cache() is mock_cache


def test_path1_does_not_call_get_service():
    mock_cache = MagicMock(name="AssistantCache")

    class InjectedWithRegistry(AssistantCacheMixin):
        def __init__(self, cache):
            self._assistant_cache = cache
            self._get_service_called = False

        def _get_service(self, cls):
            self._get_service_called = True
            return MagicMock()

    obj = InjectedWithRegistry(cache=mock_cache)
    _ = obj.assistant_cache
    assert not obj._get_service_called


# ---------------------------------------------------------------------------
# Path 2 — ServiceRegistryMixin._get_service()
# ---------------------------------------------------------------------------


def test_path2_service_registry_used_when_no_injection():
    mock_cache = MagicMock(name="AssistantCacheFromRegistry")
    obj = ServiceRegistryClass(service_return=mock_cache)
    # No _assistant_cache set — should fall through to path 2
    assert obj.assistant_cache is mock_cache


def test_path2_passes_class_not_string_to_get_service():
    """_get_service must receive the AssistantCache class, not the string 'AssistantCache'."""
    received_args = []

    class CapturingRegistry(AssistantCacheMixin):
        def _get_service(self, cls):
            received_args.append(cls)
            return MagicMock()

    obj = CapturingRegistry()
    _ = obj.assistant_cache

    assert len(received_args) == 1
    assert received_args[0].__name__ == "AssistantCache"
    assert isinstance(received_args[0], type)


# ---------------------------------------------------------------------------
# Path 3 — lazy construction from self.redis
# ---------------------------------------------------------------------------


def test_path3_constructs_from_redis():
    mock_redis = MagicMock(name="Redis")
    mock_cache_instance = MagicMock(name="AssistantCache")

    # AssistantCache is imported locally inside the property — patch at its
    # source module, not at the mixin module where it has no module-level name.
    with patch(
        "entities_api.cache.assistant_cache.AssistantCache",
        return_value=mock_cache_instance,
    ) as MockCache:
        obj = RedisClass(redis=mock_redis)
        result = obj.assistant_cache

    MockCache.assert_called_once_with(redis=mock_redis)
    assert result is mock_cache_instance


def test_path3_caches_constructed_instance():
    """Path 3 should set self._assistant_cache so subsequent accesses hit path 1."""
    mock_redis = MagicMock(name="Redis")
    mock_cache_instance = MagicMock(name="AssistantCache")

    with patch(
        "entities_api.cache.assistant_cache.AssistantCache",
        return_value=mock_cache_instance,
    ):
        obj = RedisClass(redis=mock_redis)
        first = obj.assistant_cache
        second = obj.assistant_cache

    assert first is second
    assert obj._assistant_cache is mock_cache_instance


# ---------------------------------------------------------------------------
# Path 4 — hard ValueError
# ---------------------------------------------------------------------------


def test_path4_raises_value_error_when_nothing_available():
    obj = BareClass()
    with pytest.raises(ValueError, match="AssistantCache could not be resolved"):
        _ = obj.assistant_cache


def test_path4_error_message_includes_class_name():
    obj = BareClass()
    with pytest.raises(ValueError) as exc_info:
        _ = obj.assistant_cache
    assert "BareClass" in str(exc_info.value)


def test_path4_get_assistant_cache_also_raises():
    obj = BareClass()
    with pytest.raises(ValueError):
        obj.get_assistant_cache()


# ---------------------------------------------------------------------------
# Setter — direct assignment bypasses resolution chain
# ---------------------------------------------------------------------------


def test_setter_assigns_backing_attribute():
    obj = BareClass()
    mock_cache = MagicMock(name="AssistantCache")
    obj.assistant_cache = mock_cache
    assert obj._assistant_cache is mock_cache


def test_setter_makes_subsequent_access_hit_path1():
    obj = BareClass()
    mock_cache = MagicMock(name="AssistantCache")
    obj.assistant_cache = mock_cache
    # Should now resolve via path 1 without raising
    assert obj.assistant_cache is mock_cache


# ---------------------------------------------------------------------------
# Resolution priority — path 1 beats path 2 beats path 3
# ---------------------------------------------------------------------------


def test_path1_beats_path2():
    """Injected cache takes priority over service registry."""
    injected_cache = MagicMock(name="InjectedCache")
    registry_cache = MagicMock(name="RegistryCache")

    class BothPaths(AssistantCacheMixin):
        def __init__(self):
            self._assistant_cache = injected_cache

        def _get_service(self, cls):
            return registry_cache

    obj = BothPaths()
    assert obj.assistant_cache is injected_cache


def test_path2_beats_path3():
    """Service registry takes priority over redis construction."""
    registry_cache = MagicMock(name="RegistryCache")
    mock_redis = MagicMock(name="Redis")

    class RegistryAndRedis(AssistantCacheMixin):
        def __init__(self):
            self.redis = mock_redis

        def _get_service(self, cls):
            return registry_cache

    obj = RegistryAndRedis()
    assert obj.assistant_cache is registry_cache
