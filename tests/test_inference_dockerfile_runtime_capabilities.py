from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = REPO_ROOT / "docker" / "inference" / "Dockerfile"

WORKER_SOURCE = REPO_ROOT / "src" / "api" / "training" / "inference_worker.py"

CAPABILITY_SOURCE = REPO_ROOT / "src" / "api" / "training" / "runtime_capabilities.py"


def test_inference_image_packages_runtime_capability_module():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "COPY src/api/training/inference_worker.py " "/app/inference_worker.py"
    ) in dockerfile

    assert (
        "COPY src/api/training/runtime_capabilities.py "
        "/app/src/api/training/runtime_capabilities.py"
    ) in dockerfile


def test_packaged_module_matches_worker_import_contract():
    worker = WORKER_SOURCE.read_text(encoding="utf-8")

    assert CAPABILITY_SOURCE.is_file()

    assert (
        "from src.api.training.runtime_capabilities "
        "import capture_runtime_capabilities"
    ) in worker

    assert ("_RUNTIME_CAPABILITIES_ROUTE = " '"/runtime-capabilities"') in worker
