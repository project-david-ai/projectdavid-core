"""Conservative execution-ledger reconciliation.

No scheduler, no new schema, no Assistant deletion, no nonterminal mutation.
The caller explicitly selects dry-run or ledger-only apply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from projectdavid_orm.projectdavid_orm.execution_ledger import (
    ExecutionActor,
    RunExecutionEpoch,
)
from sqlalchemy import select

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Assistant, Run
from src.api.entities_api.services.execution_ledger_service import (
    ExecutionLedgerService,
)

TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})
ACTIVE_EPOCH = frozenset({"reserved", "running"})
ACTIVE_ACTOR = frozenset({"reserved", "active"})


def state(value):
    return str(getattr(value, "value", value))


@dataclass(frozen=True)
class Finding:
    actor_id: str
    run_id: str
    epoch_id: str
    kind: str
    run_status: str
    epoch_status: str
    actor_status: str
    assistant_id: Optional[str]
    assistant_deleted: Optional[bool]
    age_seconds: int
    reason: str = ""


def classify(
    run_status,
    epoch_status,
    actor_status,
    assistant_id,
    assistant_deleted,
    age_seconds,
    stale_seconds=3600,
):
    """Pure classification: stale is a diagnostic only, never a deletion trigger."""
    if run_status == "missing":
        return "INCONSISTENT", "run missing"
    if run_status in TERMINAL:
        if epoch_status in TERMINAL and epoch_status != run_status:
            return "INCONSISTENT", "terminal outcome disagreement"
        if epoch_status not in TERMINAL | ACTIVE_EPOCH:
            return "INCONSISTENT", "unknown epoch state"
        if actor_status not in ACTIVE_ACTOR | {"retired"}:
            return "INCONSISTENT", "unknown actor state"
        if assistant_id and assistant_deleted is None:
            return "INCONSISTENT", "Assistant snapshot not found"
        if epoch_status != run_status or actor_status != "retired":
            return "TERMINAL_LEDGER_REPAIR", "incomplete ledger finalization"
        if assistant_id and assistant_deleted is False:
            return (
                "ASSISTANT_DELETION_PENDING",
                "requires separately guarded soft-delete",
            )
        return "CLEAN", ""
    if epoch_status in TERMINAL or actor_status == "retired":
        return "INCONSISTENT", "nonterminal Run has terminal ledger state"
    if epoch_status not in ACTIVE_EPOCH or actor_status not in ACTIVE_ACTOR:
        return "INCONSISTENT", "unknown nonterminal ledger state"
    if run_status == "action_required":
        return "AWAITING_USER", "SDK/user handoff; never age-expire"
    if age_seconds >= stale_seconds:
        return "SUSPECT_NONTERMINAL", "inactivity alone cannot prove abandonment"
    return "HEALTHY", ""


class ExecutionLifecycleReconciler:
    """Terminal ledger repair; Assistant cleanup is intentionally not included."""

    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory
        self.ledger = ExecutionLedgerService()

    @staticmethod
    def _finding(db, actor_id: str, stale_seconds: int, lock: bool = False):
        # Read immutable IDs first. Acquire locks Run -> epoch -> Actor.
        references = db.execute(
            select(ExecutionActor.epoch_id, RunExecutionEpoch.run_id)
            .join(RunExecutionEpoch, RunExecutionEpoch.id == ExecutionActor.epoch_id)
            .where(ExecutionActor.id == actor_id)
        ).one_or_none()
        if references is None:
            raise ValueError(f"Missing Actor or epoch: {actor_id}")
        epoch_id, run_id = references

        def get(model, row_id):
            stmt = select(model).where(model.id == row_id)
            if lock:
                stmt = stmt.with_for_update()
            return db.execute(stmt).scalar_one_or_none()

        run = get(Run, run_id)
        epoch = get(RunExecutionEpoch, epoch_id)
        actor = get(ExecutionActor, actor_id)
        if not epoch or not actor:
            raise ValueError("Execution lineage missing during inspection")
        if actor.epoch_id != epoch_id or epoch.run_id != run_id:
            raise ValueError("Execution lineage changed")
        if not actor.ephemeral:
            raise ValueError("Refusing non-ephemeral Actor")
        run_status = state(run.status) if run else "missing"
        epoch_status = state(epoch.state)
        actor_status = state(actor.state)
        assistant_id = actor.assistant_id_snapshot
        assistant = get(Assistant, assistant_id) if assistant_id else None
        assistant_deleted = (
            None
            if assistant_id and assistant is None
            else bool(assistant.deleted_at is not None) if assistant else None
        )
        last_seen = epoch.last_seen_at or epoch.created_at
        age = max(0, int(time.time()) - int(last_seen or 0))
        kind, reason = classify(
            run_status,
            epoch_status,
            actor_status,
            assistant_id,
            assistant_deleted,
            age,
            stale_seconds,
        )
        if run and epoch.owner_user_id != run.user_id:
            kind, reason = "INCONSISTENT", "epoch ownership mismatch"
        if assistant and run and assistant.owner_id != run.user_id:
            kind, reason = "INCONSISTENT", "Assistant ownership mismatch"
        return Finding(
            actor_id,
            run_id,
            epoch_id,
            kind,
            run_status,
            epoch_status,
            actor_status,
            assistant_id,
            assistant_deleted,
            age,
            reason,
        )

    def scan(self, stale_seconds: int = 3600):
        with self.session_factory() as db:
            ids = (
                db.execute(
                    select(ExecutionActor.id)
                    .where(ExecutionActor.ephemeral.is_(True))
                    .order_by(ExecutionActor.reserved_at, ExecutionActor.id)
                )
                .scalars()
                .all()
            )
        result = []
        for actor_id in ids:
            try:
                with self.session_factory() as db:
                    result.append(self._finding(db, actor_id, stale_seconds))
            except Exception as exc:
                result.append(
                    Finding(
                        actor_id, "", "", "ERROR", "", "", "", None, None, 0, repr(exc)
                    )
                )
        return result

    def reconcile_terminal(self, actor_id: str, *, dry_run: bool = True):
        """Repair only terminal ledger records; deletion and leases are out of scope."""
        if dry_run:
            with self.session_factory() as db:
                return self._finding(db, actor_id, 3600)

        with self.session_factory.begin() as db:
            current = self._finding(db, actor_id, 3600, lock=True)
            if current.kind != "TERMINAL_LEDGER_REPAIR":
                return current
            # Both canonical Run status and epoch status were checked above.
            # The native ledger methods maintain epoch -> actor lock ordering.
            if current.actor_status != "retired":
                self.ledger.retire_actor(db, actor_id=actor_id)
            if current.epoch_status != current.run_status:
                self.ledger.complete_epoch(
                    db,
                    epoch_id=current.epoch_id,
                    terminal_status=current.run_status,
                )
        with self.session_factory() as db:
            result = self._finding(db, actor_id, 3600)
        if result.kind == "TERMINAL_LEDGER_REPAIR":
            raise RuntimeError("Ledger reconciliation incomplete")
        return result

    def inspect_abandoned(self, stale_seconds: int = 3600):
        """Returns suspects only. Never changes Run or Actor state."""
        return [
            f
            for f in self.scan(stale_seconds)
            if f.kind in {"SUSPECT_NONTERMINAL", "INCONSISTENT", "ERROR"}
        ]
