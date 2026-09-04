from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.training.services.registry_service import (
    RegistryService,
    normalize_local_model_endpoint,
)

RUNTIME_ENDPOINT = "/opt/projectdavid/model-hub/models/" "model-a/variant-a/revision-a"


class FakeQuery:
    def __init__(
        self,
        existing,
    ):
        self.existing = existing

    def filter(
        self,
        *_args,
        **_kwargs,
    ):
        return self

    def first(self):
        return self.existing


class FakeSession:
    def __init__(
        self,
        existing=None,
    ):
        self.existing = existing
        self.added = []
        self.commits = 0
        self.refreshes = []

    def query(
        self,
        _model,
    ):
        return FakeQuery(self.existing)

    def add(
        self,
        value,
    ):
        self.added.append(value)

    def commit(self):
        self.commits += 1

    def refresh(
        self,
        value,
    ):
        self.refreshes.append(value)


def test_local_endpoint_accepts_runtime_child():
    assert normalize_local_model_endpoint(RUNTIME_ENDPOINT) == RUNTIME_ENDPOINT


@pytest.mark.parametrize(
    "endpoint",
    [
        "/etc/passwd",
        "/opt/projectdavid/model-hub/models",
        ("/opt/projectdavid/model-hub/models/" "../outside"),
        ("/opt/projectdavid/model-hub/models/" "model-a//variant-a"),
        r"C:\models\unsafe",
        "relative/model",
    ],
)
def test_local_endpoint_rejects_untrusted_paths(
    endpoint,
):
    with pytest.raises(HTTPException) as exc:
        normalize_local_model_endpoint(endpoint)

    assert exc.value.status_code == 422


def test_local_registration_is_idempotent_by_endpoint():
    existing = SimpleNamespace(
        id="bm_existing",
        endpoint=RUNTIME_ENDPOINT,
    )

    session = FakeSession(existing=existing)

    service = RegistryService(session)

    result = service.register_local_base_model(
        model_endpoint=RUNTIME_ENDPOINT,
        name="Model A",
    )

    assert result is existing
    assert session.added == []
    assert session.commits == 0


def test_local_registration_preserves_locator_without_hf_reinterpretation():
    session = FakeSession()

    service = RegistryService(session)

    result = service.register_local_base_model(
        model_endpoint=RUNTIME_ENDPOINT,
        name="Model A",
        family="test",
    )

    assert result.endpoint == RUNTIME_ENDPOINT

    assert result.name == "Model A"
    assert result.family == "test"

    assert session.added == [result]

    assert session.commits == 1

    assert session.refreshes == [result]
