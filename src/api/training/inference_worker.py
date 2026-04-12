"""
inference_worker.py

Sovereign Forge — Inference Worker (Ray HEAD Node or Worker Node)

This container is the Ray HEAD node by default. It owns the GPU, runs Ray Serve,
and hosts the InferenceReconciler.

When RAY_ADDRESS is set, it joins an existing Ray cluster as a worker node,
retrying until the HEAD is reachable (e.g. while Tailscale is connecting).

Architecture:
    - Starts Ray HEAD node (or joins existing cluster if RAY_ADDRESS is set)
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

Hyperparam resolution priority (applies to all vLLM engine args):
    1. InferenceDeployment DB column  — set at activation time via API
    2. VLLM_DEFAULT_* env var         — node-level override in docker-compose.yml
    3. Built-in safe default          — coded at module level below

NODE_IP notes:
    When NODE_IP is set, Ray advertises that IP address to the cluster instead
    of auto-detecting the network interface. This is required for Tailscale
    deployments where each node has a stable 100.x.x.x Tailscale IP that is
    mutually reachable by all cluster members.

    Set NODE_IP to the Tailscale IP on both HEAD and worker nodes:
        HEAD   : NODE_IP=<head-tailscale-ip>  (in .env or docker-compose.yml)
        WORKER : NODE_IP=<worker-tailscale-ip> (in RunPod template env vars)

    When NODE_IP is set on the HEAD, set RAY_ADDRESS on the worker to:
        RAY_ADDRESS=ray://<head-tailscale-ip>:10001
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


from projectdavid_common import LoggingUtility

logging_utility = LoggingUtility()
log = logging_utility

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAY_DASHBOARD_PORT = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))
POLL_INTERVAL = int(os.getenv("INFERENCE_POLL_INTERVAL", "20"))
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")
SERVE_HTTP_PORT = int(os.getenv("SERVE_HTTP_PORT", "8000"))

# Tailscale / overlay network IP for this node.
# When set, Ray advertises this IP instead of auto-detecting the interface.
# Required for Tailscale-based multi-node clusters.
NODE_IP = os.getenv("NODE_IP") or None

# Retry config for worker node cluster join
RAY_JOIN_MAX_RETRIES = int(os.getenv("RAY_JOIN_MAX_RETRIES", "20"))
RAY_JOIN_RETRY_DELAY = int(os.getenv("RAY_JOIN_RETRY_DELAY", "15"))

# ---------------------------------------------------------------------------
# Node-level vLLM defaults
# These are the fallback values when no DB column override is present.
# Override via environment variables in docker-compose.yml.
# ---------------------------------------------------------------------------

# Fraction of GPU VRAM vLLM may allocate for weights + KV cache.
# 0.50 is conservative — safe on 8 GiB cards without profiling.
_DEFAULT_GPU_MEM_UTIL = float(os.getenv("VLLM_DEFAULT_GPU_MEM_UTIL", "0.50"))
_GPU_MEM_UTIL_MIN = 0.10
_GPU_MEM_UTIL_MAX = 0.95

# Maximum sequence length (prompt + completion tokens).
# Larger values consume more KV cache VRAM linearly.
_DEFAULT_MAX_MODEL_LEN = int(os.getenv("VLLM_DEFAULT_MAX_MODEL_LEN", "4096"))


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
    Reserves 1 GPU via ray_actor_options.

    Supports:
      - Base model inference
      - LoRA adapter inference (via enable_lora + lora_request)
      - Sovereign Forge raw completions API

    SSE chunks are returned in /v1/completions format (choices[0].text)
    so _http_stream in VLLMRawStream can consume them without adaptation.

    All vLLM engine hyperparams are passed in at deploy time from the
    InferenceDeployment DB record. None values fall back to vLLM defaults.
    """

    def __init__(
        self,
        model_endpoint: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = _DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = _DEFAULT_GPU_MEM_UTIL,
        lora_modules: dict = None,
        # --- Per-deployment vLLM engine hyperparam overrides ---
        # Read from InferenceDeployment DB columns at deploy time.
        # None values defer to vLLM's own defaults except where noted.
        quantization: str = None,  # "awq", "awq_marlin", "gptq", "bitsandbytes", None = full precision
        dtype: str = None,  # "float16", "bfloat16", "auto", None → defaults to float16 below
        enforce_eager: bool = False,  # True = disable CUDA graphs (slower but useful for OOM debugging)
        limit_mm_per_prompt: dict = None,  # e.g. {"image": 2, "video": 0} — caps vision token counts per request
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest

        self.model_endpoint = model_endpoint
        self.lora_modules = lora_modules or {}

        log.info(
            "🚀 VLLMDeployment initialising — model=%s tp=%d gpu_mem_util=%.2f "
            "max_model_len=%d quantization=%s dtype=%s enforce_eager=%s lora_modules=%s",
            model_endpoint,
            tensor_parallel_size,
            gpu_memory_utilization,
            max_model_len,
            quantization,
            dtype,
            enforce_eager,
            list(self.lora_modules.keys()),
        )

        # Build engine kwargs — only pass optional params when explicitly set
        # to avoid overriding vLLM's internal defaults with None values.
        engine_kwargs = dict(
            model=model_endpoint,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=bool(self.lora_modules),
            max_loras=max(1, len(self.lora_modules)),
            enforce_eager=enforce_eager,
            # dtype: float16 is the safe explicit default for mixed-precision GPUs.
            # None would let vLLM auto-select which can pick bfloat16 on some cards
            # causing a cast warning — we prefer explicit control.
            dtype=dtype if dtype is not None else "float16",
        )

        # quantization: only pass when set — None means full precision and
        # passing it explicitly would override model config auto-detection.
        if quantization is not None:
            engine_kwargs["quantization"] = quantization

        # limit_mm_per_prompt: per-modality token cap for vision models.
        # Prevents runaway token counts from high-res images on small GPUs.
        if limit_mm_per_prompt is not None:
            engine_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        engine_args = AsyncEngineArgs(**engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

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
        """
        Handle raw completions requests from the Sovereign Forge path.

        Accepts either:
          - "prompt"   : pre-rendered prompt string (from _stream_sovereign_forge)
          - "messages" : ChatML messages array (fallback for direct callers)

        SSE chunks are returned in /v1/completions format:
            choices[0].text   ← delta text
        This matches what _http_stream in VLLMRawStream expects.

        LoRA lookup: the model field may be either the fine_tuned_model_id
        (ftm_...) or the deployment ID (vllm_dep_...). If no matching LoRA
        request is found by name and exactly one adapter is loaded, the
        loaded adapter is used automatically.
        """
        import json as _json

        from starlette.responses import StreamingResponse
        from vllm import SamplingParams

        body = await request.json()
        messages = body.get("messages", [])
        model_name = body.get("model", self.model_endpoint)
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        prompt = body.get("prompt") or self._format_messages(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        lora_request = self._lora_requests.get(model_name)
        if lora_request is None and len(self._lora_requests) == 1:
            lora_request = next(iter(self._lora_requests.values()))
            log.debug(
                "VLLMDeployment: model field '%s' did not match a LoRA key — "
                "using sole loaded adapter '%s'.",
                model_name,
                next(iter(self._lora_requests.keys())),
            )

        request_id = f"req_{int(time.time() * 1000)}"

        if stream:

            async def _event_stream():
                prev_text = ""
                async for output in self.engine.generate(
                    prompt,
                    sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                ):
                    if output.outputs:
                        full_text = output.outputs[0].text
                        delta = full_text[len(prev_text) :]
                        prev_text = full_text
                        if delta:
                            chunk = {
                                "choices": [
                                    {
                                        "text": delta,
                                        "finish_reason": None,
                                    }
                                ]
                            }
                            yield f"data: {_json.dumps(chunk)}\n\n"

                final = {"choices": [{"text": "", "finish_reason": "stop"}]}
                yield f"data: {_json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_event_stream(), media_type="text/event-stream")

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
            "object": "text_completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": output_text,
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
        """Format ChatML messages into a prompt string (fallback path)."""
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
        """
        Resolve a base_model_id to its HuggingFace endpoint string.

        Raises RuntimeError if the BaseModel record no longer exists.
        A missing record means the deployment references a deregistered model
        and should not proceed.
        """
        base = (
            db.query(self._BaseModel)
            .filter(self._BaseModel.id == base_model_id)
            .first()
        )
        if not base:
            raise RuntimeError(
                f"BaseModel '{base_model_id}' not found in registry. "
                f"The InferenceDeployment references a model that no longer exists. "
                f"Deregister this deployment or re-register the base model."
            )
        return base.endpoint

    def _build_lora_modules(self, dep) -> dict:
        if not dep.fine_tuned_model_id:
            return {}
        if not dep.fine_tuned_model or not dep.fine_tuned_model.storage_path:
            return {}
        adapter_path = f"{SHARED_PATH}/{dep.fine_tuned_model.storage_path}"
        return {dep.fine_tuned_model_id: adapter_path}

    def _deploy(self, db: Session, dep) -> None:
        """Create a Ray Serve deployment on this HEAD node.

        All vLLM engine hyperparams are resolved from the InferenceDeployment
        DB record first, falling back to env vars, then built-in safe defaults.
        This means each model can have its own tuned configuration without
        requiring a container rebuild or compose change.
        """
        from projectdavid_common.schemas.enums import StatusEnum

        deployment_name = _deployment_name(dep.id)
        model_endpoint = self._resolve_model_endpoint(db, dep.base_model_id)
        lora_modules = self._build_lora_modules(dep)

        # --- Resolve all hyperparams from DB columns → env vars → defaults ---

        # Number of GPUs for tensor parallelism (1 = single GPU, no sharding)
        tp_size = getattr(dep, "tensor_parallel_size", 1) or 1

        # GPU VRAM fraction — clamped to [0.10, 0.95] by _resolve helper
        gpu_mem_util = _resolve_gpu_memory_utilization(dep)

        # Max sequence length: DB column → VLLM_DEFAULT_MAX_MODEL_LEN env → 4096
        max_model_len = getattr(dep, "max_model_len", None) or _DEFAULT_MAX_MODEL_LEN

        # Quantization: "awq", "awq_marlin", "gptq", "bitsandbytes", or None
        quantization = getattr(dep, "quantization", None)

        # Compute dtype: "float16", "bfloat16", "auto", or None → VLLMDeployment defaults to float16
        dtype = getattr(dep, "dtype", None)

        # Disable CUDA graphs — slower but useful when debugging OOM crashes
        enforce_eager = bool(getattr(dep, "enforce_eager", False))

        # Per-modality token cap — critical for vision models on constrained GPUs
        # e.g. {"image": 2} prevents a single high-res image from consuming all KV cache
        limit_mm_per_prompt = getattr(dep, "limit_mm_per_prompt", None)

        log.info(
            "🚢 Deploying via Ray Serve: %s model=%s tp=%d gpu_mem_util=%.2f "
            "max_model_len=%d quantization=%s dtype=%s enforce_eager=%s lora=%s",
            deployment_name,
            model_endpoint,
            tp_size,
            gpu_mem_util,
            max_model_len,
            quantization,
            dtype,
            enforce_eager,
            list(lora_modules.keys()),
        )

        bound = VLLMDeployment.options(
            name=deployment_name,
            ray_actor_options={"num_gpus": tp_size},
        ).bind(
            model_endpoint=model_endpoint,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            lora_modules=lora_modules,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
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
            "✅ Ray Serve deployment active: %s (gpu_mem_util=%.2f max_model_len=%d)",
            deployment_name,
            gpu_mem_util,
            max_model_len,
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
            expected_serve_names = {_deployment_name(dep_id) for dep_id in db_ids}
            serve_names = self._get_active_serve_deployments()

            for dep in deployments:
                deployment_name = _deployment_name(dep.id)
                if deployment_name not in serve_names:
                    log.warning(
                        "🚨 Deployment drift — %s not in Ray Serve. Redeploying.",
                        deployment_name,
                    )
                    try:
                        self._deploy(db, dep)
                    except RuntimeError as e:
                        log.error("Skipping deployment %s: %s", deployment_name, e)

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
    # ── Phase 1: Start Ray (HEAD or Worker) ──────────────────────────────
    #
    # RAY_ADDRESS unset → HEAD node.
    #   node_ip_address passed when NODE_IP is set so the HEAD advertises
    #   its Tailscale IP to workers rather than the Docker bridge IP.
    #   Required for Tailscale-based multi-node clusters.
    #
    # RAY_ADDRESS set → Worker node joining existing cluster.
    #   node_ip_address passed when NODE_IP is set so the worker registers
    #   its Tailscale IP with the cluster, enabling bidirectional Ray
    #   communication without an SSH tunnel for Ray traffic.
    #   Retries up to RAY_JOIN_MAX_RETRIES times with RAY_JOIN_RETRY_DELAY
    #   seconds between attempts — tolerates Tailscale connect delay and
    #   transient HEAD unavailability without crashing the container.

    ray_address = os.getenv("RAY_ADDRESS") or None

    # Set RAY_NODE_IP_ADDRESS so Ray advertises the Tailscale IP
    # instead of auto-detecting the Docker bridge interface.
    if NODE_IP:
        os.environ["RAY_NODE_IP_ADDRESS"] = NODE_IP
        log.info("🌐 Node IP override: %s (Tailscale)", NODE_IP)

    if ray_address:
        # ── Worker node ───────────────────────────────────────────────────
        log.info("🔗 Joining Ray cluster at %s", ray_address)
        for attempt in range(1, RAY_JOIN_MAX_RETRIES + 1):
            try:
                ray.init(
                    address=ray_address,
                    ignore_reinit_error=True,
                    logging_level="WARNING",
                )
                log.info(
                    "✅ Joined Ray cluster — resources: %s", ray.cluster_resources()
                )
                break
            except Exception as e:
                log.warning(
                    "⏳ Ray cluster not ready (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    RAY_JOIN_MAX_RETRIES,
                    e,
                    RAY_JOIN_RETRY_DELAY,
                )
                if attempt == RAY_JOIN_MAX_RETRIES:
                    log.error(
                        "❌ Could not join Ray cluster after %d attempts. Exiting.",
                        RAY_JOIN_MAX_RETRIES,
                    )
                    raise
                time.sleep(RAY_JOIN_RETRY_DELAY)
    else:
        # ── HEAD node ─────────────────────────────────────────────────────
        ray.init(
            address=None,
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host="0.0.0.0",  # nosec B104
            dashboard_port=RAY_DASHBOARD_PORT,
            logging_level="WARNING",
        )

        log.info(
            "🌐 Ray HEAD started — dashboard: http://localhost:%d", RAY_DASHBOARD_PORT
        )
        log.info("🔵 Ray resources: %s", ray.cluster_resources())

    # ── Phase 2: Start Ray Serve ──────────────────────────────────────────
    serve.start(
        detached=True,
        http_options={"host": "0.0.0.0", "port": SERVE_HTTP_PORT},  # nosec B104
    )
    log.info("🎯 Ray Serve started on port %d", SERVE_HTTP_PORT)

    # ── Phase 3: Reconciliation loop ──────────────────────────────────────
    reconciler = InferenceReconciler()
    log.info("👀 InferenceReconciler active — polling every %ds", POLL_INTERVAL)

    while True:
        reconciler.reconcile()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
