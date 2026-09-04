"""
Runtime capability collection for the Project David inference environment.

This module intentionally has no Ray or vLLM imports at module import time.
It is safe to unit-test without starting the inference stack.

Important semantic distinction:
    torch.version.cuda reports the CUDA version PyTorch was built against.
    It is NOT treated as the loaded CUDA runtime version.

The returned payload is the schema consumed by Q Model Hub.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable

SCHEMA_VERSION = 1


class RuntimeCapabilityProbeError(RuntimeError):
    """Raised when authoritative runtime capability collection cannot complete."""


def _clean_required_string(value: Any, field: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise RuntimeCapabilityProbeError(f"{field} is unavailable")
    return rendered


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    rendered = str(value).strip()
    return rendered or None


def _resolve_distribution_version(
    distribution: str,
    version_getter: Callable[[str], str],
) -> str:
    try:
        value = version_getter(distribution)
    except Exception as exc:
        raise RuntimeCapabilityProbeError(
            f"version metadata unavailable for {distribution}"
        ) from exc

    return _clean_required_string(
        value,
        f"{distribution} version",
    )


def _resolve_torch_version(
    torch_module: Any,
    version_getter: Callable[[str], str],
) -> str:
    value = getattr(torch_module, "__version__", None)
    if value:
        return _clean_required_string(value, "torch version")

    return _resolve_distribution_version("torch", version_getter)


def _resolve_cudnn_version(torch_module: Any) -> str | None:
    try:
        cudnn = getattr(
            getattr(torch_module, "backends", None),
            "cudnn",
            None,
        )
        if cudnn is None:
            return None

        value = cudnn.version()
    except Exception:
        return None

    return _optional_string(value)


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeCapabilityProbeError("PyTorch runtime is unavailable") from exc

    return torch


def _accelerator_identity(
    *,
    accelerator_api: str,
    index: int,
    properties: Any,
) -> str:
    raw_uuid = getattr(properties, "uuid", None)

    if raw_uuid:
        rendered = str(raw_uuid).strip()
        if rendered:
            return rendered

    return f"{accelerator_api}:{index}"


def _compute_capability(
    *,
    accelerator_api: str,
    properties: Any,
) -> str | None:
    if accelerator_api != "cuda":
        return None

    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)

    if major is None or minor is None:
        return None

    return f"{int(major)}.{int(minor)}"


def _visible_accelerators(
    *,
    torch_module: Any,
    accelerator_api: str,
) -> list[dict[str, Any]]:
    if accelerator_api == "cpu":
        return []

    try:
        count = int(torch_module.cuda.device_count())
    except Exception as exc:
        raise RuntimeCapabilityProbeError(
            "could not enumerate runtime-visible accelerators"
        ) from exc

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    vendor = "amd" if accelerator_api == "rocm" else "nvidia"

    for index in range(count):
        try:
            properties = torch_module.cuda.get_device_properties(index)
        except Exception as exc:
            raise RuntimeCapabilityProbeError(
                f"could not inspect accelerator {index}"
            ) from exc

        accelerator_id = _accelerator_identity(
            accelerator_api=accelerator_api,
            index=index,
            properties=properties,
        )

        if accelerator_id in seen_ids:
            raise RuntimeCapabilityProbeError(
                f"duplicate runtime accelerator id: {accelerator_id}"
            )

        seen_ids.add(accelerator_id)

        total_memory = getattr(properties, "total_memory", 0)

        try:
            total_vram_bytes = int(total_memory)
        except (TypeError, ValueError) as exc:
            raise RuntimeCapabilityProbeError(
                f"invalid VRAM value for accelerator {index}"
            ) from exc

        if total_vram_bytes < 0:
            raise RuntimeCapabilityProbeError(
                f"negative VRAM value for accelerator {index}"
            )

        result.append(
            {
                "id": accelerator_id,
                "vendor": vendor,
                "model": _optional_string(getattr(properties, "name", None)),
                "compute_capability": _compute_capability(
                    accelerator_api=accelerator_api,
                    properties=properties,
                ),
                "total_vram_bytes": total_vram_bytes,
            }
        )

    return result


def capture_runtime_capabilities(
    *,
    project_david_version: str,
    torch_module: Any | None = None,
    version_getter: Callable[[str], str] = metadata.version,
) -> dict[str, Any]:
    """
    Capture authoritative capabilities of the inference-worker environment.

    project_david_version is explicit rather than guessed from an unrelated
    installed SDK distribution. The inference-worker wiring supplies the
    authoritative Core build version.

    The snapshot is captured before model-specific Ray actors are created.
    """

    project_version = _clean_required_string(
        project_david_version,
        "Project David version",
    )

    torch_runtime = torch_module if torch_module is not None else _load_torch()

    backend_version = _resolve_distribution_version(
        "vllm",
        version_getter,
    )

    torch_version = _resolve_torch_version(
        torch_runtime,
        version_getter,
    )

    torch_version_namespace = getattr(
        torch_runtime,
        "version",
        None,
    )

    torch_cuda_build = _optional_string(
        getattr(
            torch_version_namespace,
            "cuda",
            None,
        )
    )

    torch_rocm_build = _optional_string(
        getattr(
            torch_version_namespace,
            "hip",
            None,
        )
    )

    try:
        accelerator_available = bool(torch_runtime.cuda.is_available())
    except Exception as exc:
        raise RuntimeCapabilityProbeError(
            "could not determine accelerator availability"
        ) from exc

    if not accelerator_available:
        accelerator_api = "cpu"
    elif torch_rocm_build:
        accelerator_api = "rocm"
    else:
        accelerator_api = "cuda"

    visible_accelerators = _visible_accelerators(
        torch_module=torch_runtime,
        accelerator_api=accelerator_api,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "project_david_version": project_version,
        "backend": {
            "id": "vllm",
            "version": backend_version,
        },
        "runtime": {
            "accelerator_api": accelerator_api,
            # Deliberately null until queried from the actual runtime API.
            # torch.version.cuda is the PyTorch build CUDA version and must
            # never be misreported as cuda_runtime_version.
            "cuda_runtime_version": None,
            # Same rule for ROCm: torch.version.hip is build metadata.
            "rocm_runtime_version": None,
        },
        "frameworks": {
            "torch": {
                "version": torch_version,
                "cuda_version": torch_cuda_build,
                "cudnn_version": _resolve_cudnn_version(torch_runtime),
            }
        },
        "visible_accelerators": visible_accelerators,
    }
