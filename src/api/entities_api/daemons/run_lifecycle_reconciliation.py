# src/api/entities_api/daemons/run_lifecycle_reconciliation.py
"""Read-only by default. --apply repairs terminal LEDGER state only."""
import argparse
from collections import Counter

from src.api.entities_api.services.execution_lifecycle_reconciler import (
    ExecutionLifecycleReconciler,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Retire terminal Actors and finalize terminal epochs only",
    )
    parser.add_argument("--stale-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive")

    reconciler = ExecutionLifecycleReconciler()
    findings = reconciler.scan(stale_seconds=args.stale_seconds)
    counts = Counter(f.kind for f in findings)
    print("MODE:", "APPLY (ledger only)" if args.apply else "DRY RUN")
    for kind in sorted(counts):
        print(f"{kind}: {counts[kind]}")
    for finding in findings:
        if finding.kind in {"CLEAN", "HEALTHY", "AWAITING_USER"}:
            continue
        print(
            f"{finding.kind} actor={finding.actor_id} run={finding.run_id} "
            f"assistant={finding.assistant_id} reason={finding.reason}"
        )

    if not args.apply:
        print("DRY RUN COMPLETE - NO CHANGES")
        return

    repairs = 0
    for finding in findings:
        if finding.kind != "TERMINAL_LEDGER_REPAIR":
            continue
        result = reconciler.reconcile_terminal(finding.actor_id, dry_run=False)
        if result.kind in {"CLEAN", "ASSISTANT_DELETION_PENDING"}:
            repairs += 1
        else:
            print(f"REVIEW REQUIRED {finding.actor_id}: {result.kind}")
    print(f"LEDGER REPAIRS: {repairs}")
    print("ASSISTANT DELETION IS NOT AUTOMATED")


if __name__ == "__main__":
    main()
