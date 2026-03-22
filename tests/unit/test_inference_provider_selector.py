from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.entities_api.orchestration.engine.inference_provider_selector import (
    TOP_LEVEL_ROUTING_MAP, InferenceProviderSelector)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_arbiter():
    return MagicMock(name="InferenceArbiter")


@pytest.fixture
def selector(mock_arbiter):
    return InferenceProviderSelector(arbiter=mock_arbiter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_handlers():
    """
    Patch every handler class in TOP_LEVEL_ROUTING_MAP so no real
    __init__ is called during tests.
    """
    patches = {}
    for prefix, handler_cls in TOP_LEVEL_ROUTING_MAP.items():
        module = handler_cls.__module__
        name = handler_cls.__name__
        mock = MagicMock(name=name)
        mock.__name__ = name
        patches[prefix] = (f"{module}.{name}", mock)
    return patches


# ---------------------------------------------------------------------------
# Routing — known prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected_prefix",
    [
        ("hyperbolic/meta-llama/Llama-3.3-70B-Instruct", "hyperbolic/"),
        ("together-ai/Qwen/Qwen3-235B-A22B", "together-ai/"),
        ("ollama/llama3.2", "ollama/"),
        ("vllm/mistralai/Mistral-7B-Instruct-v0.3", "vllm/"),
        # Case insensitive
        ("HYPERBOLIC/some-model", "hyperbolic/"),
        ("TOGETHER-AI/some-model", "together-ai/"),
    ],
)
def test_select_provider_routes_known_prefix(selector, model_id, expected_prefix):
    expected_class = TOP_LEVEL_ROUTING_MAP[expected_prefix]
    with patch.object(
        selector,
        "_get_or_create_general_handler",
        return_value=MagicMock(name="handler"),
    ) as mock_get:
        instance, _ = selector.select_provider(model_id)
        called_with = mock_get.call_args[0][0]
        assert called_with is expected_class


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------


def test_select_provider_returns_mapped_model_name(selector):
    """If model_id is in MODEL_MAP, api_model_name should be the mapped value."""
    from src.api.entities_api.orchestration.engine.inference_provider_selector import \
        MODEL_MAP

    # Find a model_id that is actually in MODEL_MAP and has a known prefix
    mappable = {
        k: v
        for k, v in MODEL_MAP.items()
        if any(k.lower().startswith(p) for p in TOP_LEVEL_ROUTING_MAP)
    }
    if not mappable:
        pytest.skip("No mappable model found in MODEL_MAP for current routing prefixes")

    model_id, expected_api_name = next(iter(mappable.items()))

    with patch.object(
        selector,
        "_get_or_create_general_handler",
        return_value=MagicMock(),
    ):
        _, api_name = selector.select_provider(model_id)
        assert api_name == expected_api_name


def test_select_provider_falls_back_to_model_id_when_not_in_map(selector):
    """If model_id is not in MODEL_MAP, api_model_name should equal model_id."""
    model_id = "ollama/some-obscure-model-not-in-map"
    with patch.object(
        selector,
        "_get_or_create_general_handler",
        return_value=MagicMock(),
    ):
        _, api_name = selector.select_provider(model_id)
        assert api_name == model_id


# ---------------------------------------------------------------------------
# Unknown prefix — raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_model_id",
    [
        "openai/gpt-4o",
        "anthropic/claude-3-opus",
        "groq/llama3-70b",
        "completely-unknown/model",
        "",
        "no-slash-at-all",
    ],
)
def test_select_provider_raises_for_unknown_prefix(selector, bad_model_id):
    with pytest.raises(ValueError, match="Invalid or unknown model identifier prefix"):
        selector.select_provider(bad_model_id)


# ---------------------------------------------------------------------------
# Handler caching — same instance returned on second call
# ---------------------------------------------------------------------------


def test_handler_is_cached_across_calls(mock_arbiter):
    """
    _get_or_create_general_handler should return the same instance
    on repeated calls for the same handler class.
    """
    selector = InferenceProviderSelector(arbiter=mock_arbiter)
    handler_cls = TOP_LEVEL_ROUTING_MAP["ollama/"]

    mock_instance = MagicMock(name="OllamaHandlerInstance")

    with patch.object(handler_cls, "__init__", return_value=None):
        with patch(
            f"{handler_cls.__module__}.{handler_cls.__name__}",
            return_value=mock_instance,
        ):
            # Prime the cache manually
            selector._general_handler_cache[handler_cls.__name__] = mock_instance

            instance_1 = selector._get_or_create_general_handler(handler_cls)
            instance_2 = selector._get_or_create_general_handler(handler_cls)

    assert instance_1 is instance_2


# ---------------------------------------------------------------------------
# Handler instantiation failure — raises ValueError
# ---------------------------------------------------------------------------


def test_select_provider_raises_on_handler_instantiation_failure(mock_arbiter):
    selector = InferenceProviderSelector(arbiter=mock_arbiter)

    failing_class = MagicMock(name="FailingHandler")
    failing_class.__name__ = "FailingHandler"
    failing_class.side_effect = RuntimeError("boom")

    with patch.dict(
        "src.api.entities_api.orchestration.engine.inference_provider_selector.TOP_LEVEL_ROUTING_MAP",
        {"failing/": failing_class},
    ):
        selector._sorted_routing_keys = ["failing/"]
        with pytest.raises(ValueError, match="Instantiation failed for handler"):
            selector._get_or_create_general_handler(failing_class)


# ---------------------------------------------------------------------------
# Prefix sort order — longer prefix wins over shorter
# ---------------------------------------------------------------------------


def test_sorted_routing_keys_longest_first(selector):
    """
    Longer prefixes must be checked first to avoid partial matches.
    e.g. 'together-ai/' must not shadow a hypothetical 'together-ai/special/' prefix.
    """
    keys = selector._sorted_routing_keys
    for i in range(len(keys) - 1):
        assert len(keys[i]) >= len(
            keys[i + 1]
        ), f"Routing key '{keys[i]}' should be >= length of '{keys[i + 1]}'"
