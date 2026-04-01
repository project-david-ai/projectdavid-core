"""
inference_worker.py

Sovereign Forge — Inference Worker (Ray HEAD Node)

This container is the Ray HEAD node. It owns the GPU, runs Ray Serve,
and hosts the InferenceReconciler. training_worker joins as a Ray WORKER
via ray://inference_worker:10001.

Because this is the HEAD node, Ray Serve actors default to this node —
where vLLM is installed. No node pinning required.

Architecture:
    - Starts Ray HEAD node
    - Starts Ray client server on port 10001 (for training_worker to join)
    - Polls inference_deployments DB table every poll_interval seconds
    - Reconciles Ray Serve deployments with DB state

GPU memory notes:
    vLLM runs a full dummy forward pass (profile_run) during initialisation to
    size the KV cache.  On small GPUs (≤ 8 GiB) a gpu_memory_utilization of
    0.85 (the vLLM default) leaves insufficient headroom for the sampler's
    top-k/top-p sort and causes an OOM before any real inference begins.

    The safe default here is 0.50.  Per-deployment overrides are read from the
    DB column `gpu_memory_utilization` when present, with that value clamped to
    [0.10, 0.95] to guard against bad data.  Set VLLM_DEFAULT_GPU_MEM_UTIL in
    the environment to change the default for all deployments on this node.
"""

import logging
import os
import socket
import time

import ray
from ray import serve
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAY_DASHBOARD_PORT = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))
RAY_CLIENT_SERVER_PORT = int(os.getenv("RAY_CLIENT_SERVER_PORT", "10001"))
POLL_INTERVAL = int(os.getenv("INFERENCE_POLL_INTERVAL", "20"))
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")
SERVE_HTTP_PORT = int(os.getenv("SERVE_HTTP_PORT", "8000"))

# Safe default for gpu_memory_utilization.
# vLLM's own default (0.90) is too aggressive for ≤ 8 GiB cards.
# Override per-node with VLLM_DEFAULT_GPU_MEM_UTIL.
_DEFAULT_GPU_MEM_UTIL = float(os.getenv("VLLM_DEFAULT_GPU_MEM_UTIL", "0.50"))
_GPU_MEM_UTIL_MIN = 0.10
_GPU_MEM_UTIL_MAX = 0.95


def _resolve_gpu_memory_utilization(dep) -> float:
    """
    Return the gpu_memory_utilization to use for this deployment.

    Priority:
      1. dep.gpu_memory_utilization  (DB column, if present and valid)
      2. VLLM_DEFAULT_GPU_MEM_UTIL   (node-level env override)
      3. 0.50                        (built-in safe default)

    The value is clamped to [0.10, 0.95] regardless of source.
    """
    raw = getattr(dep, "gpu_memory_utilization", None)
    if raw is not None:
        try:
            value = float(raw)
            clamped = max(_GPU_MEM_UTIL_MIN, min(_GPU_MEM_UTIL_MAX, value))
            if clamped != value:
                log.warning(
                    "gpu_memory_utilization %.2f for deployment %s clamped to %.2f",
                    value,
                    dep.id,
                    clamped,
                )
            return clamped
        except (TypeError, ValueError):
            log.warning(
                "Invalid gpu_memory_utilization %r for deployment %s — using default %.2f",
                raw,
                dep.id,
                _DEFAULT_GPU_MEM_UTIL,
            )
    return _DEFAULT_GPU_MEM_UTIL


# ---------------------------------------------------------------------------
# Ray Serve deployment — vLLM wrapped as a serve application
# ---------------------------------------------------------------------------


@serve.deployment(
    name="vllm_inference",
    ray_actor_options={"num_gpus": 1},
    health_check_period_s=30,
    health_check_timeout_s=60,
)
class VLLMDeployment:
    """
    Ray Serve deployment wrapping a vLLM AsyncLLMEngine.

    Runs on the inference_worker HEAD node where vLLM is installed.
    Reserves 1 GPU via ray_actor_options — Ray's scheduler prevents
    training jobs from claiming this GPU while inference is active.

    Supports:
      - Base model inference
      - LoRA adapter inference (via enable_lora + lora_request)
      - OpenAI-compatible chat completions API

    gpu_memory_utilization defaults to 0.50 (safe for 8 GiB cards).
    Pass a higher value only when you have profiled headroom on your
    target hardware.  Values above 0.85 risk OOM during vLLM's
    internal profile_run on small GPUs.
    """

    def __init__(
        self,
        model_endpoint: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = _DEFAULT_GPU_MEM_UTIL,
        lora_modules: dict = None,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest

        self.model_endpoint = model_endpoint
        self.lora_modules = lora_modules or {}

        log.info(
            "🚀 VLLMDeployment initialising — model=%s tp=%d gpu_mem_util=%.2f lora_modules=%s",
            model_endpoint,
            tensor_parallel_size,
            gpu_memory_utilization,
            list(self.lora_modules.keys()),
        )

        engine_args = AsyncEngineArgs(
            model=model_endpoint,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=bool(self.lora_modules),
            max_loras=max(1, len(self.lora_modules)),
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Use lora_path (lora_local_path is deprecated in vLLM 0.8.x+)
        self._lora_requests = {
            name: LoRARequest(
                lora_name=name,
                lora_int_id=idx + 1,
                lora_path=path,
            )
            for idx, (name, path) in enumerate(self.lora_modules.items())
        }

        log.info("✅ VLLMDeployment ready — model=%s", model_endpoint)

    async def __call__(self, request):
        """Handle OpenAI-compatible chat completion requests."""
        from vllm import SamplingParams

        body = await request.json()
        messages = body.get("messages", [])
        model_name = body.get("model", self.model_endpoint)
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)

        prompt = self._format_messages(messages)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        lora_request = self._lora_requests.get(model_name)
        request_id = f"req_{int(time.time() * 1000)}"
        output_text = ""

        async for output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            if output.outputs:
                output_text = output.outputs[0].text

        return {
            "id": request_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output_text.split()),
                "total_tokens": len(prompt.split()) + len(output_text.split()),
            },
        }

    def _format_messages(self, messages: list) -> str:
        """Format ChatML messages into a prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def check_health(self):
        return True


# ---------------------------------------------------------------------------
# Deployment name helper
# ---------------------------------------------------------------------------


def _deployment_name(deployment_id: str) -> str:
    """Convert dep_... ID to a valid Ray Serve deployment name.

    Ray Serve names must be alphanumeric + underscore.
    dep_DWtgOyY5iTpkV1gfslIUwJ → vllm_dep_DWtgOyY5iTpkV1gfslIUwJ

    Note: we only replace hyphens — the ID itself never contains hyphens
    (projectdavid IDs use alphanumeric + underscore already). This makes
    the name→ID reverse mapping unambiguous.
    """
    return f"vllm_{deployment_id.replace('-', '_')}"


# ---------------------------------------------------------------------------
# Reconciliation loop
# ---------------------------------------------------------------------------


class InferenceReconciler:
    """
    Polls the inference_deployments DB table and reconciles Ray Serve state.

    Pending deployments  → create Ray Serve deployment (GPU reserved)
    Active deployments   → verify healthy
    Orphaned deployments → delete Ray Serve deployment (GPU released)
    """

    def __init__(self):
        from src.api.training.db.database import SessionLocal
        from src.api.training.models.models import BaseModel, InferenceDeployment

        self._SessionLocal = SessionLocal
        self._InferenceDeployment = InferenceDeployment
        self._BaseModel = BaseModel
        # Maps deployment DB ID → Ray Serve app name
        # e.g. "dep_DWtgOyY5iTpkV1gfslIUwJ" → "vllm_dep_DWtgOyY5iTpkV1gfslIUwJ"
        self._active: dict[str, str] = {}

    def _get_db(self) -> Session:
        return self._SessionLocal()

    def _get_pending_deployments(self, db: Session) -> list:
        from projectdavid_common.schemas.enums import StatusEnum

        return (
            db.query(self._InferenceDeployment)
            .filter(
                self._InferenceDeployment.node_id.is_not(None),
                self._InferenceDeployment.status.in_(
                    [StatusEnum.pending, StatusEnum.active]
                ),
            )
            .all()
        )

    def _resolve_model_endpoint(self, db: Session, base_model_id: str) -> str:
        base = (
            db.query(self._BaseModel)
            .filter(self._BaseModel.id == base_model_id)
            .first()
        )
        return base.endpoint if base else base_model_id

    def _build_lora_modules(self, dep) -> dict:
        if not dep.fine_tuned_model_id:
            return {}
        if not dep.fine_tuned_model or not dep.fine_tuned_model.storage_path:
            return {}
        adapter_path = f"{SHARED_PATH}/{dep.fine_tuned_model.storage_path}"
        return {dep.fine_tuned_model_id: adapter_path}

    def _deploy(self, db: Session, dep) -> None:
        """Create a Ray Serve deployment on this HEAD node."""
        from projectdavid_common.schemas.enums import StatusEnum

        deployment_name = _deployment_name(dep.id)
        model_endpoint = self._resolve_model_endpoint(db, dep.base_model_id)
        lora_modules = self._build_lora_modules(dep)
        tp_size = getattr(dep, "tensor_parallel_size", 1) or 1

        # Resolve gpu_memory_utilization: DB record → env override → safe default.
        # This is the primary fix for OOM during vLLM's profile_run on small GPUs.
        gpu_mem_util = _resolve_gpu_memory_utilization(dep)

        log.info(
            "🚢 Deploying via Ray Serve: %s model=%s tp=%d gpu_mem_util=%.2f lora=%s",
            deployment_name,
            model_endpoint,
            tp_size,
            gpu_mem_util,
            list(lora_modules.keys()),
        )

        bound = VLLMDeployment.options(
            name=deployment_name,
            ray_actor_options={"num_gpus": tp_size},
        ).bind(
            model_endpoint=model_endpoint,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem_util,
            lora_modules=lora_modules,
        )

        serve.run(
            bound,
            name=deployment_name,
            route_prefix=f"/{deployment_name}",
            blocking=False,
        )

        dep.status = StatusEnum.active
        dep.internal_hostname = f"http://inference_worker:8000/{deployment_name}"
        db.commit()

        self._active[dep.id] = deployment_name
        log.info(
            "✅ Ray Serve deployment active: %s (gpu_mem_util=%.2f)",
            deployment_name,
            gpu_mem_util,
        )

    def _delete_deployment(self, deployment_name: str) -> None:
        try:
            serve.delete(deployment_name)
            log.info("🛑 Ray Serve deployment deleted: %s", deployment_name)
        except Exception as e:
            log.warning("Could not delete deployment %s: %s", deployment_name, e)

    def _get_active_serve_deployments(self) -> set:
        try:
            statuses = serve.status().applications
            return set(statuses.keys())
        except Exception:
            return set()

    def reconcile(self) -> None:
        db = self._get_db()
        try:
            deployments = self._get_pending_deployments(db)
            db_ids = {dep.id for dep in deployments}

            # Build the expected set of serve names from DB records
            expected_serve_names = {_deployment_name(dep_id) for dep_id in db_ids}

            serve_names = self._get_active_serve_deployments()

            # Deploy anything that should exist but doesn't
            for dep in deployments:
                deployment_name = _deployment_name(dep.id)
                if deployment_name not in serve_names:
                    log.warning(
                        "🚨 Deployment drift — %s not in Ray Serve. Redeploying.",
                        deployment_name,
                    )
                    self._deploy(db, dep)

            # Delete anything running in Ray Serve that has no DB record.
            # Compare against expected_serve_names derived from DB — NOT by
            # reconstructing IDs from serve names (which is lossy and buggy).
            for name in serve_names:
                if name.startswith("vllm_") and name not in expected_serve_names:
                    log.info("🧹 Orphaned Ray Serve deployment — deleting: %s", name)
                    self._delete_deployment(name)

        except Exception as e:
            log.error("Reconciliation error: %s", e)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    # ── Phase 1: Start Ray HEAD ───────────────────────────────────────────
    ray.init(
        address=None,
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # nosec B104
        dashboard_port=RAY_DASHBOARD_PORT,
        logging_level="WARNING",
    )
    log.info("🌐 Ray HEAD started — dashboard: http://localhost:%d", RAY_DASHBOARD_PORT)

    # ── Phase 2: Start Ray client server ─────────────────────────────────
    # Allows training_worker to join the cluster via ray://inference_worker:10001
    # ray.util.client.server.serve() is non-blocking — starts gRPC server
    # in background threads and returns immediately.
    from ray.util.client.server import serve as start_ray_client_server

    _client_server_handle = start_ray_client_server(  # noqa: F841
        f"0.0.0.0:{RAY_CLIENT_SERVER_PORT}"  # nosec B104
    )
    time.sleep(2)  # Give gRPC server time to bind
    log.info("🔌 Ray client server started on port %d", RAY_CLIENT_SERVER_PORT)
    log.info("🔵 Ray resources: %s", ray.cluster_resources())

    # ── Phase 3: Start Ray Serve ──────────────────────────────────────────
    serve.start(
        detached=True,
        http_options={"host": "0.0.0.0", "port": SERVE_HTTP_PORT},  # nosec B104
    )
    log.info("🎯 Ray Serve started on port %d", SERVE_HTTP_PORT)

    # ── Phase 4: Reconciliation loop ──────────────────────────────────────
    reconciler = InferenceReconciler()
    log.info("👀 InferenceReconciler active — polling every %ds", POLL_INTERVAL)

    while True:
        reconciler.reconcile()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
