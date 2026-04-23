# src/api/training/services/disk_preflight.py
"""
Pre-dispatch disk space preconditions for training jobs.

Training runs are long-lived (hours) and fail noisily when disk pressure
hits mid-run — SMB write errors, HuggingFace dataset cache explosions,
adapter save failures. Cheaper to reject the job at create time with a
clear error than to let the user wait 40 minutes for a storage failure.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()


@dataclass
class DiskCheckResult:
    ok: bool
    path: str
    free_gb: float
    required_gb: float
    reason: str = ""


class DiskPreflight:
    """
    Runs disk space preconditions before expensive operations.

    Checks two locations by default:
        - Samba share (adapter output, dataset staging)
        - Scratch / tmp (trainer working dir, HF cache spillover)

    Thresholds are env-configurable so operators with different hardware
    can tune without code changes.
    """

    ENV_MIN_SAMBA_FREE_GB = "MIN_SAMBA_FREE_GB"
    ENV_MIN_SCRATCH_FREE_GB = "MIN_SCRATCH_FREE_GB"
    ENV_SAMBA_PATH = "SHARED_PATH"
    ENV_SCRATCH_PATH = "SCRATCH_PATH"

    DEFAULT_MIN_SAMBA_FREE_GB = 5.0
    DEFAULT_MIN_SCRATCH_FREE_GB = 10.0
    DEFAULT_SAMBA_PATH = "/mnt/training_data"
    DEFAULT_SCRATCH_PATH = "/tmp"  # nosec B108

    def __init__(
        self,
        samba_path: str | None = None,
        scratch_path: str | None = None,
        min_samba_free_gb: float | None = None,
        min_scratch_free_gb: float | None = None,
    ) -> None:
        self.samba_path = samba_path or os.getenv(
            self.ENV_SAMBA_PATH, self.DEFAULT_SAMBA_PATH
        )
        self.scratch_path = scratch_path or os.getenv(
            self.ENV_SCRATCH_PATH, self.DEFAULT_SCRATCH_PATH
        )
        self.min_samba_free_gb = (
            min_samba_free_gb
            if min_samba_free_gb is not None
            else float(
                os.getenv(
                    self.ENV_MIN_SAMBA_FREE_GB,
                    str(self.DEFAULT_MIN_SAMBA_FREE_GB),
                )
            )
        )
        self.min_scratch_free_gb = (
            min_scratch_free_gb
            if min_scratch_free_gb is not None
            else float(
                os.getenv(
                    self.ENV_MIN_SCRATCH_FREE_GB,
                    str(self.DEFAULT_MIN_SCRATCH_FREE_GB),
                )
            )
        )

    def check_path(self, path: str, required_gb: float) -> DiskCheckResult:
        p = Path(path)
        if not p.exists():
            return DiskCheckResult(
                ok=False,
                path=path,
                free_gb=0.0,
                required_gb=required_gb,
                reason=f"Path does not exist: {path}",
            )
        try:
            free_bytes = shutil.disk_usage(path).free
            free_gb = free_bytes / (1024**3)
        except OSError as e:
            return DiskCheckResult(
                ok=False,
                path=path,
                free_gb=0.0,
                required_gb=required_gb,
                reason=f"Could not stat path ({e})",
            )

        if free_gb < required_gb:
            return DiskCheckResult(
                ok=False,
                path=path,
                free_gb=round(free_gb, 2),
                required_gb=required_gb,
                reason=f"Only {free_gb:.2f} GB free, need {required_gb:.2f} GB",
            )

        return DiskCheckResult(
            ok=True,
            path=path,
            free_gb=round(free_gb, 2),
            required_gb=required_gb,
        )

    def run(self) -> List[DiskCheckResult]:
        """Run all configured checks. Returns one result per path."""
        results = [
            self.check_path(self.samba_path, self.min_samba_free_gb),
            self.check_path(self.scratch_path, self.min_scratch_free_gb),
        ]
        self._log(results)
        return results

    @property
    def failures(self) -> List[DiskCheckResult]:
        """Convenience: run and return only the failures."""
        return [r for r in self.run() if not r.ok]

    def _log(self, results: List[DiskCheckResult]) -> None:
        for r in results:
            if r.ok:
                logging_utility.info(
                    "Disk preflight OK: %s has %.2f GB free (need %.2f GB)",
                    r.path,
                    r.free_gb,
                    r.required_gb,
                )
            else:
                logging_utility.warning(
                    "Disk preflight FAIL: %s — %s",
                    r.path,
                    r.reason,
                )
