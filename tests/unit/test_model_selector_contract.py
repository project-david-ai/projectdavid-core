from __future__ import annotations

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import Base
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import Session

from src.api.entities_api.orchestration.mixins.json_utils_mixin import JsonUtilsMixin
from src.api.entities_api.services.inference_resolver import (
    BaseModel,
    InferenceDeployment,
    InferenceResolver,
)
from src.api.training.services.deployment_service import DeploymentService

HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
BASE_MODEL_ID = "bm_XoSgMaQbOSuPnVXZ3gcQF6"
FINE_TUNED_MODEL_ID = "ftm_exampleAdapterId"
DEPLOYMENT_ID = "dep_exampleDeploymentId"
RAY_SERVE_URL = "http://inference_worker:8000/vllm_dep_exampleDeploymentId"


@pytest.mark.parametrize(
    "public_selector,provider_native_selector",
    [
        (f"vllm/{HF_MODEL_ID}", HF_MODEL_ID),
        (f"vllm/{BASE_MODEL_ID}", BASE_MODEL_ID),
        (f"vllm/{FINE_TUNED_MODEL_ID}", FINE_TUNED_MODEL_ID),
    ],
)
def test_vllm_translation_removes_exactly_one_provider_prefix(
    public_selector, provider_native_selector
):
    assert JsonUtilsMixin()._get_model_map(public_selector) == provider_native_selector


@pytest.fixture
def deployment_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            BaseModel(
                id=BASE_MODEL_ID,
                name="Qwen 2.5 VL 3B AWQ",
                family="qwen",
                parameter_count="3B",
                is_multimodal=True,
                endpoint=HF_MODEL_ID,
            )
        )
        session.add(
            InferenceDeployment(
                id=DEPLOYMENT_ID,
                node_id="node_test",
                internal_hostname=RAY_SERVE_URL,
                base_model_id=BASE_MODEL_ID,
                fine_tuned_model_id=FINE_TUNED_MODEL_ID,
                status=StatusEnum.active,
                last_seen=123,
            )
        )
        session.commit()
        yield session

    engine.dispose()


@pytest.mark.parametrize(
    "model_selector",
    [
        f"vllm/{HF_MODEL_ID}",
        f"vllm/{BASE_MODEL_ID}",
        f"vllm/{FINE_TUNED_MODEL_ID}",
        "vllm/vllm_dep_exampleDeploymentId",
    ],
)
def test_resolver_routes_all_vllm_model_identities_to_the_active_deployment(
    deployment_session, model_selector
):
    assert (
        InferenceResolver.resolve_vllm_url(deployment_session, model_selector)
        == RAY_SERVE_URL
    )


def test_resolver_ignores_inactive_deployments(deployment_session):
    deployment = deployment_session.get(InferenceDeployment, DEPLOYMENT_ID)
    deployment.status = StatusEnum.pending
    deployment_session.commit()

    assert (
        InferenceResolver.resolve_vllm_url(deployment_session, f"vllm/{BASE_MODEL_ID}")
        is None
    )


def test_fine_tuned_activation_links_ftm_to_resolved_base_model():
    db = MagicMock()
    service = DeploymentService.__new__(DeploymentService)
    service.db = db
    service.registry = MagicMock()

    fine_tuned_model = SimpleNamespace(
        id=FINE_TUNED_MODEL_ID,
        user_id="user_test",
        base_model=HF_MODEL_ID,
        is_active=False,
    )
    base_model = SimpleNamespace(id=BASE_MODEL_ID, endpoint=HF_MODEL_ID)

    service.get_fine_tuned_model = MagicMock(return_value=fine_tuned_model)
    service._get_blocking_deployments = MagicMock(return_value=[])
    service.registry.resolve.return_value = base_model
    service._find_available_node = MagicMock(return_value="node_test")
    service._check_node_capacity = MagicMock()

    with patch(
        "src.api.training.services.deployment_service."
        "IdentifierService.generate_prefixed_id",
        return_value=DEPLOYMENT_ID,
    ):
        result = service.activate_fine_tuned_model(FINE_TUNED_MODEL_ID)

    service._get_blocking_deployments.assert_called_once_with()
    service.registry.resolve.assert_called_once_with(HF_MODEL_ID)

    created_deployment = db.add.call_args.args[0]
    assert created_deployment.fine_tuned_model_id == FINE_TUNED_MODEL_ID
    assert created_deployment.base_model_id == BASE_MODEL_ID
    assert result["model_id"] == FINE_TUNED_MODEL_ID
    assert result["base_model_id"] == BASE_MODEL_ID


def _load_inference_worker_with_fakes(monkeypatch):
    fake_serve = SimpleNamespace(
        deployment=lambda **_kwargs: lambda deployment_class: deployment_class
    )
    fake_ray = types.ModuleType("ray")
    fake_ray.serve = fake_serve

    class FakeEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self):
            self.last_lora_request = None
            self.last_engine_args = None

        async def generate(
            self, engine_input, sampling_params, request_id, lora_request=None
        ):
            self.last_lora_request = lora_request
            yield SimpleNamespace(outputs=[SimpleNamespace(text="adapter output")])

    fake_engine = FakeEngine()

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            fake_engine.last_engine_args = engine_args
            return fake_engine

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeEngineArgs
    fake_vllm.AsyncLLMEngine = FakeAsyncLLMEngine
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_vllm_lora = types.ModuleType("vllm.lora")
    fake_vllm_lora_request = types.ModuleType("vllm.lora.request")
    fake_vllm_lora_request.LoRARequest = FakeLoRARequest

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_vllm_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_vllm_lora_request)
    sys.modules.pop("src.api.training.inference_worker", None)

    module = importlib.import_module("src.api.training.inference_worker")
    return module, fake_engine


def test_ftm_selector_builds_and_selects_explicit_lora_request(monkeypatch):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)
    adapter_path = f"/mnt/training_data/models/{FINE_TUNED_MODEL_ID}"
    deployment = inference_worker.VLLMDeployment(
        model_endpoint=HF_MODEL_ID,
        lora_modules={FINE_TUNED_MODEL_ID: adapter_path},
    )

    assert deployment.engine is fake_engine
    assert deployment._lora_requests[FINE_TUNED_MODEL_ID].lora_name == (
        FINE_TUNED_MODEL_ID
    )
    assert deployment._lora_requests[FINE_TUNED_MODEL_ID].lora_path == adapter_path

    class Request:
        async def json(self):
            return {
                "prompt": "test",
                "model": FINE_TUNED_MODEL_ID,
                "stream": False,
            }

    response = asyncio.run(deployment(Request()))

    assert (
        fake_engine.last_lora_request is deployment._lora_requests[FINE_TUNED_MODEL_ID]
    )
    assert response["model"] == FINE_TUNED_MODEL_ID
    assert response["choices"][0]["text"] == "adapter output"


def test_vllm_deployment_omits_dtype_when_not_explicitly_configured(monkeypatch):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)

    deployment = inference_worker.VLLMDeployment(
        model_endpoint=HF_MODEL_ID,
        dtype=None,
    )

    assert deployment.engine is fake_engine
    assert "dtype" not in fake_engine.last_engine_args.kwargs


def test_vllm_deployment_passes_explicit_dtype(monkeypatch):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)

    deployment = inference_worker.VLLMDeployment(
        model_endpoint=HF_MODEL_ID,
        dtype="bfloat16",
    )

    assert deployment.engine is fake_engine
    assert fake_engine.last_engine_args.kwargs["dtype"] == "bfloat16"


def test_kimi_k3_family_enables_trust_remote_code(monkeypatch):
    inference_worker, _ = _load_inference_worker_with_fakes(monkeypatch)

    config = inference_worker._get_vision_family_config("tiny-random/kimi-k3")

    assert config["trust_remote_code"] is True


def test_hf_kimi_k3_deployment_preserves_trust_remote_code(monkeypatch):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)

    deployment = inference_worker.VLLMDeployment(
        model_endpoint="tiny-random/kimi-k3",
    )

    assert deployment.engine is fake_engine
    assert fake_engine.last_engine_args.kwargs["trust_remote_code"] is True


def test_model_hub_local_kimi_k3_forces_trust_remote_code_false(monkeypatch):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)

    model_endpoint = "/opt/projectdavid/model-hub/models/" "model-a/kimi-k3/revision-a"

    deployment = inference_worker.VLLMDeployment(
        model_endpoint=model_endpoint,
    )

    assert deployment.engine is fake_engine
    assert fake_engine.last_engine_args.kwargs["trust_remote_code"] is False


def test_model_hub_namespace_forces_false_even_for_noncanonical_descendant(
    monkeypatch,
):
    inference_worker, fake_engine = _load_inference_worker_with_fakes(monkeypatch)

    model_endpoint = "/opt/projectdavid/model-hub/models/" "../kimi-k3"

    deployment = inference_worker.VLLMDeployment(
        model_endpoint=model_endpoint,
    )

    assert deployment.engine is fake_engine
    assert fake_engine.last_engine_args.kwargs["trust_remote_code"] is False
