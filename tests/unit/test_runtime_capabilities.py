from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.training.runtime_capabilities import (
    RuntimeCapabilityProbeError,
    capture_runtime_capabilities,
)


class _FakeCudnn:
    def __init__(self, value):
        self._value = value

    def version(self):
        return self._value


class _FakeCuda:
    def __init__(self, *, available, devices):
        self._available = available
        self._devices = devices

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._devices)

    def get_device_properties(self, index):
        return self._devices[index]


def _fake_torch(
    *,
    available=True,
    devices=None,
    cuda_build="12.8",
    hip_build=None,
    cudnn=91002,
):
    return SimpleNamespace(
        __version__="2.7.1+cu128",
        version=SimpleNamespace(
            cuda=cuda_build,
            hip=hip_build,
        ),
        cuda=_FakeCuda(
            available=available,
            devices=devices or [],
        ),
        backends=SimpleNamespace(
            cudnn=_FakeCudnn(cudnn),
        ),
    )


def _version_getter(name):
    versions = {
        "vllm": "0.10.1",
        "torch": "2.7.1",
    }
    return versions[name]


def test_cuda_runtime_snapshot():
    torch = _fake_torch(
        devices=[
            SimpleNamespace(
                uuid="GPU-test-123",
                name="NVIDIA Test GPU",
                total_memory=8 * 1024**3,
                major=8,
                minor=9,
            )
        ]
    )

    result = capture_runtime_capabilities(
        project_david_version="1.47.1",
        torch_module=torch,
        version_getter=_version_getter,
    )

    assert result == {
        "schema_version": 1,
        "project_david_version": "1.47.1",
        "backend": {
            "id": "vllm",
            "version": "0.10.1",
        },
        "runtime": {
            "accelerator_api": "cuda",
            "cuda_runtime_version": None,
            "rocm_runtime_version": None,
        },
        "frameworks": {
            "torch": {
                "version": "2.7.1+cu128",
                "cuda_version": "12.8",
                "cudnn_version": "91002",
            }
        },
        "visible_accelerators": [
            {
                "id": "GPU-test-123",
                "vendor": "nvidia",
                "model": "NVIDIA Test GPU",
                "compute_capability": "8.9",
                "total_vram_bytes": 8 * 1024**3,
            }
        ],
    }


def test_cpu_only_fallback():
    torch = _fake_torch(
        available=False,
        devices=[],
        cuda_build=None,
        cudnn=None,
    )

    result = capture_runtime_capabilities(
        project_david_version="1.47.1",
        torch_module=torch,
        version_getter=_version_getter,
    )

    assert result["runtime"]["accelerator_api"] == "cpu"
    assert result["visible_accelerators"] == []
    assert result["frameworks"]["torch"]["cuda_version"] is None
    assert result["frameworks"]["torch"]["cudnn_version"] is None


def test_rocm_is_not_misreported_as_cuda():
    torch = _fake_torch(
        available=True,
        hip_build="6.3",
        cuda_build=None,
        devices=[
            SimpleNamespace(
                uuid=None,
                name="AMD Test GPU",
                total_memory=16 * 1024**3,
                major=0,
                minor=0,
            )
        ],
    )

    result = capture_runtime_capabilities(
        project_david_version="1.47.1",
        torch_module=torch,
        version_getter=_version_getter,
    )

    assert result["runtime"]["accelerator_api"] == "rocm"
    assert result["runtime"]["rocm_runtime_version"] is None
    assert result["visible_accelerators"][0]["vendor"] == "amd"
    assert result["visible_accelerators"][0]["id"] == "rocm:0"
    assert result["visible_accelerators"][0]["compute_capability"] is None


def test_torch_build_cuda_is_not_reported_as_runtime_cuda():
    torch = _fake_torch(
        devices=[
            SimpleNamespace(
                uuid="GPU-test",
                name="GPU",
                total_memory=1,
                major=8,
                minor=9,
            )
        ],
        cuda_build="12.8",
    )

    result = capture_runtime_capabilities(
        project_david_version="1.47.1",
        torch_module=torch,
        version_getter=_version_getter,
    )

    assert result["frameworks"]["torch"]["cuda_version"] == "12.8"
    assert result["runtime"]["cuda_runtime_version"] is None


def test_missing_project_david_version_fails_closed():
    with pytest.raises(
        RuntimeCapabilityProbeError,
        match="Project David version",
    ):
        capture_runtime_capabilities(
            project_david_version="",
            torch_module=_fake_torch(
                available=False,
            ),
            version_getter=_version_getter,
        )


def test_missing_vllm_version_fails_closed():
    def missing_version(_name):
        raise LookupError("missing")

    with pytest.raises(
        RuntimeCapabilityProbeError,
        match="vllm",
    ):
        capture_runtime_capabilities(
            project_david_version="1.47.1",
            torch_module=_fake_torch(
                available=False,
            ),
            version_getter=missing_version,
        )


def test_duplicate_accelerator_ids_fail_closed():
    torch = _fake_torch(
        devices=[
            SimpleNamespace(
                uuid="GPU-duplicate",
                name="GPU 0",
                total_memory=1,
                major=8,
                minor=9,
            ),
            SimpleNamespace(
                uuid="GPU-duplicate",
                name="GPU 1",
                total_memory=1,
                major=8,
                minor=9,
            ),
        ]
    )

    with pytest.raises(
        RuntimeCapabilityProbeError,
        match="duplicate runtime accelerator id",
    ):
        capture_runtime_capabilities(
            project_david_version="1.47.1",
            torch_module=torch,
            version_getter=_version_getter,
        )
