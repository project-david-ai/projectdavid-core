"""
Durable execution-lineage service.

Owns transactional database operations for:
    - Run execution epochs
    - Supervisor and worker actor reservations
    - Actor identity snapshots
    - Parent/child Run delegations
    - Structured, hash-chained execution events
    - Epoch and actor lifecycle transitions

Uses the released projectdavid-orm execution-ledger models.

Transaction contract
--------------------
Every method accepts an existing SQLAlchemy Session.

The CALLER owns commit/rollback. This service never commits or
opens an independent transaction. This allows lineage writes to
participate in a larger atomic operation.

This is an internal trusted service, not an HTTP authorization layer.

Do not persist prompts, reasoning, credentials, request bodies,
raw tool output, or arbitrary exception text in event details.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any, Mapping, Optional

from fastapi import HTTPException
from projectdavid_orm.projectdavid_orm.execution_ledger import (
    EvidenceRetentionHold,
    ExecutionActor,
    ExecutionEvent,
    RunDelegation,
    RunExecutionEpoch,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import Action, Run
from src.api.entities_api.services.erasure_state import require_mutable_user


class ExecutionLedgerError(RuntimeError):
    """Base class for execution-ledger contract violations."""


class LedgerNotFound(ExecutionLedgerError):
    """A required Run, Action, epoch, or actor was not found."""


class LedgerConflict(ExecutionLedgerError):
    """An operation conflicts with existing durable lineage."""


class LedgerValidationError(ExecutionLedgerError):
    """Invalid ledger input or unsafe evidence metadata."""


class ExecutionLedgerService:
    """
    Transactional persistence service for agent execution lineage.

    Lock ordering:
        Admission prefix:    User -> erasure receipt
        reserve_epoch:       prefix -> Run -> root Run (if distinct) -> latest epoch
        reserve_actor:       prefix -> epoch -> actor
        activate_actor:      prefix -> epoch -> actor
        link_delegation:     prefix -> parent epoch -> child epoch
        append_event:        epoch -> latest event
        complete_epoch:      epoch
        retire_actor:        epoch -> actor

    Callers coordinating multiple operations must preserve a
    consistent lock ordering.

    MySQL/InnoDB is the intended concurrency engine.
    """

    ACTIVE_EPOCH_STATES = frozenset({"reserved", "running"})
    TERMINAL_EPOCH_STATES = frozenset({"completed", "failed", "cancelled", "expired"})

    RETENTION_DAYS = {
        "structured": 365,
        "tool_output": 30,
        "security": 180,
    }

    SAFE_EVENT_DETAIL_KEYS = frozenset(
        {
            "status",
            "outcome",
            "reason_code",
            "error_code",
            "error_type",
            "tool_name",
            "provider",
            "model",
            "delegation_id",
            "checkpoint_id",
            "artifact_kind",
            "resource_type",
            "http_status",
            "duration_ms",
            "byte_count",
            "attempt",
        }
    )

    @staticmethod
    def _now_seconds() -> int:
        return int(time.time())

    @staticmethod
    def _now_microseconds() -> int:
        return time.time_ns() // 1_000

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _validate_identifier(
        value: str,
        field: str,
        maximum: int = 64,
    ) -> str:
        if not isinstance(value, str):
            raise LedgerValidationError(f"{field} must be a string")

        if not value or len(value) > maximum:
            raise LedgerValidationError(f"{field} must contain 1-{maximum} characters")

        if any(ord(char) < 32 for char in value):
            raise LedgerValidationError(f"{field} contains a control character")

        return value

    @classmethod
    def _validate_code(
        cls,
        value: str,
        field: str,
        maximum: int,
    ) -> str:
        cls._validate_identifier(value, field, maximum)

        if re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
            raise LedgerValidationError(f"{field} must be a machine-readable code")

        return value

    @classmethod
    def _safe_details(
        cls,
        details: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if details is None:
            return {}

        if not isinstance(details, Mapping):
            raise LedgerValidationError("Event details must be a mapping")

        result: dict[str, Any] = {}

        for key, value in details.items():
            if key not in cls.SAFE_EVENT_DETAIL_KEYS:
                raise LedgerValidationError(
                    f"Event detail field is not permitted: {key!r}"
                )

            if not isinstance(value, (str, int, float, bool)):
                raise LedgerValidationError(f"Event detail {key!r} must be scalar")

            if isinstance(value, str):
                if len(value) > 128:
                    raise LedgerValidationError(
                        f"Event detail {key!r} exceeds 128 characters"
                    )

                if any(ord(char) < 32 for char in value):
                    raise LedgerValidationError(
                        f"Event detail {key!r} contains control characters"
                    )

            result[key] = value

        try:
            json.dumps(
                result,
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise LedgerValidationError("Event details are not valid JSON") from exc

        return result

    @staticmethod
    def _require_admissible_owner(db: Session, owner_id: Optional[str]) -> None:
        """Fence new work against quarantine for the caller's transaction."""
        if not owner_id:
            raise LedgerConflict("Execution has no owner")
        try:
            require_mutable_user(db, owner_id)
        except HTTPException as exc:
            if exc.status_code in (404, 409):
                raise LedgerConflict(
                    "Execution owner is unavailable for admission"
                ) from exc
            raise

    @staticmethod
    def _require_run(
        db: Session,
        run_id: str,
        *,
        lock: bool = False,
    ) -> Run:
        stmt = select(Run).where(Run.id == run_id)

        if lock:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)

        run = db.execute(stmt).scalar_one_or_none()

        if run is None:
            raise LedgerNotFound(f"Run not found: {run_id}")

        return run

    @staticmethod
    def _require_epoch(
        db: Session,
        epoch_id: str,
        *,
        lock: bool = False,
    ) -> RunExecutionEpoch:
        stmt = select(RunExecutionEpoch).where(RunExecutionEpoch.id == epoch_id)

        if lock:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)

        epoch = db.execute(stmt).scalar_one_or_none()

        if epoch is None:
            raise LedgerNotFound(f"Execution epoch not found: {epoch_id}")

        return epoch

    @staticmethod
    def _require_actor(
        db: Session,
        actor_id: str,
        *,
        lock: bool = False,
    ) -> ExecutionActor:
        stmt = select(ExecutionActor).where(ExecutionActor.id == actor_id)

        if lock:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)

        actor = db.execute(stmt).scalar_one_or_none()

        if actor is None:
            raise LedgerNotFound(f"Execution actor not found: {actor_id}")

        return actor

    def reserve_epoch(
        self,
        db: Session,
        *,
        run_id: str,
        root_run_id: Optional[str] = None,
        new_attempt: bool = False,
    ) -> RunExecutionEpoch:
        """
        Return the current active epoch or create the first one.

        Locking the Run row serializes concurrent reservations.

        A terminal epoch cannot accidentally be reused. Opening
        another epoch requires new_attempt=True.

        For root execution:
            run_id == root_run_id

        For delegated execution:
            run_id != root_run_id
        """
        self._validate_identifier(run_id, "run_id")
        root_id = root_run_id or run_id
        self._validate_identifier(root_id, "root_run_id")

        owner_id = self._require_run(db, run_id).user_id
        self._require_admissible_owner(db, owner_id)
        run = self._require_run(db, run_id, lock=True)
        if run.user_id != owner_id:
            raise LedgerConflict("Run ownership changed during admission")

        if root_id == run_id:
            root = run
        else:
            # Reject a foreign root before acquiring its row lock.
            root = self._require_run(db, root_id)
            if root.user_id != owner_id:
                raise LedgerConflict(
                    "Delegated Run and root Run belong to different users"
                )
            root = self._require_run(db, root_id, lock=True)

        if run.user_id != root.user_id:
            raise LedgerConflict("Delegated Run and root Run belong to different users")

        stmt = (
            select(RunExecutionEpoch)
            .where(RunExecutionEpoch.run_id == run_id)
            .order_by(RunExecutionEpoch.epoch_number.desc())
            .limit(1)
            .with_for_update()
            .execution_options(populate_existing=True)
        )

        latest = db.execute(stmt).scalar_one_or_none()

        if latest is not None:
            if latest.owner_user_id != owner_id:
                raise LedgerConflict("Existing execution epoch has a different owner")
            if latest.root_run_id != root_id:
                raise LedgerConflict(
                    "Existing execution epoch has a different root Run"
                )

            if latest.state in self.ACTIVE_EPOCH_STATES:
                return latest

            if not new_attempt:
                raise LedgerConflict(
                    "Latest execution epoch is terminal; "
                    "an explicit new_attempt is required"
                )

            next_number = latest.epoch_number + 1
        else:
            next_number = 1

        now = self._now_seconds()

        epoch = RunExecutionEpoch(
            id=self._new_id(),
            run_id=run_id,
            root_run_id=root_id,
            owner_user_id=root.user_id,
            epoch_number=next_number,
            state="reserved",
            created_at=now,
            last_seen_at=now,
        )

        db.add(epoch)
        db.flush()

        return epoch

    def reserve_actor(
        self,
        db: Session,
        *,
        epoch_id: str,
        slot: str,
        execution_role: str,
        ephemeral: bool = True,
    ) -> ExecutionActor:
        """
        Reserve one actor for an epoch/slot.

        Call BEFORE creating the ephemeral Assistant.

        The epoch lock serializes reservations. The database
        UNIQUE(epoch_id, slot) constraint is the final guard.

        Repeated calls return the same reservation.
        """
        self._validate_identifier(epoch_id, "epoch_id")
        self._validate_code(slot, "slot", 48)
        self._validate_code(
            execution_role,
            "execution_role",
            48,
        )

        owner_id = self._require_epoch(db, epoch_id).owner_user_id
        self._require_admissible_owner(db, owner_id)
        epoch = self._require_epoch(db, epoch_id, lock=True)
        if epoch.owner_user_id != owner_id:
            raise LedgerConflict("Epoch ownership changed during admission")

        if epoch.state not in self.ACTIVE_EPOCH_STATES:
            raise LedgerConflict("Cannot reserve an actor on a terminal epoch")

        stmt = (
            select(ExecutionActor)
            .where(
                ExecutionActor.epoch_id == epoch_id,
                ExecutionActor.slot == slot,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )

        existing = db.execute(stmt).scalar_one_or_none()

        if existing is not None:
            if existing.execution_role != execution_role:
                raise LedgerConflict("Actor slot already belongs to a different role")

            if bool(existing.ephemeral) != bool(ephemeral):
                raise LedgerConflict("Actor slot has a different lifecycle policy")

            if existing.state == "retired":
                raise LedgerConflict(
                    "Actor slot has been retired; " "start a new execution epoch"
                )

            return existing

        actor = ExecutionActor(
            id=self._new_id(),
            epoch_id=epoch_id,
            slot=slot,
            execution_role=execution_role,
            assistant_id_snapshot=None,
            model_snapshot=None,
            ephemeral=ephemeral,
            state="reserved",
            reserved_at=self._now_seconds(),
        )

        db.add(actor)
        db.flush()

        return actor

    def activate_actor(
        self,
        db: Session,
        *,
        actor_id: str,
        assistant_id: str,
        model: Optional[str] = None,
    ) -> ExecutionActor:
        """
        Attach the actual Assistant identity to a reservation.

        The snapshot is write-once: another assistant_id cannot
        silently replace it.

        Assistant deletion does not erase this snapshot.

        Lock order: epoch -> actor.

        Activation is only permitted while the epoch remains active.
        The first activation transitions the epoch to running.
        """
        self._validate_identifier(actor_id, "actor_id")
        self._validate_identifier(
            assistant_id,
            "assistant_id",
        )

        if model is not None:
            self._validate_identifier(
                model,
                "model",
                128,
            )

        # Read only the immutable epoch reference.
        # Do not acquire the actor row lock before the epoch lock.
        epoch_id = db.execute(
            select(ExecutionActor.epoch_id).where(ExecutionActor.id == actor_id)
        ).scalar_one_or_none()

        if epoch_id is None:
            raise LedgerNotFound(f"Execution actor not found: {actor_id}")

        # Account admission precedes the epoch and actor locks.
        owner_id = self._require_epoch(db, epoch_id).owner_user_id
        self._require_admissible_owner(db, owner_id)
        epoch = self._require_epoch(db, epoch_id, lock=True)
        if epoch.owner_user_id != owner_id:
            raise LedgerConflict("Epoch ownership changed during admission")

        actor = self._require_actor(
            db,
            actor_id,
            lock=True,
        )

        if actor.epoch_id != epoch_id:
            raise LedgerConflict("Actor epoch changed during admission")

        if epoch.state not in self.ACTIVE_EPOCH_STATES:
            raise LedgerConflict("Cannot activate an actor on a terminal epoch")

        if actor.state == "retired":
            raise LedgerConflict("Cannot reactivate a retired actor")

        if (
            actor.assistant_id_snapshot is not None
            and actor.assistant_id_snapshot != assistant_id
        ):
            raise LedgerConflict("Actor already belongs to another Assistant")

        if (
            model is not None
            and actor.model_snapshot is not None
            and actor.model_snapshot != model
        ):
            raise LedgerConflict("Actor already has a different model snapshot")

        actor.assistant_id_snapshot = assistant_id

        if model is not None:
            actor.model_snapshot = model

        actor.state = "active"

        now = self._now_seconds()

        if actor.activated_at is None:
            actor.activated_at = now

        # Start the epoch only on first activation.
        # Keep started_at stable on idempotent reuse.
        if epoch.state == "reserved":
            epoch.state = "running"

            if epoch.started_at is None:
                epoch.started_at = now

        epoch.last_seen_at = now

        db.flush()
        return actor

    def retire_actor(
        self,
        db: Session,
        *,
        actor_id: str,
    ) -> ExecutionActor:
        """
        Retire an actor without removing its identity snapshot.

        This does not delete the Assistant itself. Assistant
        lifecycle cleanup remains a separate responsibility.

        Lock order: epoch -> actor.
        """
        # Read the immutable reference before acquiring row locks.
        epoch_id = db.execute(
            select(ExecutionActor.epoch_id).where(ExecutionActor.id == actor_id)
        ).scalar_one_or_none()

        if epoch_id is None:
            raise LedgerNotFound(f"Execution actor not found: {actor_id}")

        # Preserve the global lock order.
        self._require_epoch(
            db,
            epoch_id,
            lock=True,
        )

        actor = self._require_actor(
            db,
            actor_id,
            lock=True,
        )

        if actor.state != "retired":
            actor.state = "retired"
            actor.retired_at = self._now_seconds()
            db.flush()

        return actor

    def link_delegation(
        self,
        db: Session,
        *,
        parent_epoch_id: str,
        child_epoch_id: str,
        parent_action_id: str,
        initiating_actor_id: Optional[str] = None,
    ) -> RunDelegation:
        """
        Persist the parent Action -> child execution relationship.

        Idempotent when called with the same relationship.

        Requires:
            - Different parent and child Runs
            - Same root Run
            - Parent Action belongs to the parent Run
            - Initiating actor, if supplied, belongs to parent epoch
        """
        owner_id = self._require_epoch(db, parent_epoch_id).owner_user_id
        if self._require_epoch(db, child_epoch_id).owner_user_id != owner_id:
            raise LedgerConflict("Delegation epochs belong to different owners")
        self._require_admissible_owner(db, owner_id)
        parent = self._require_epoch(db, parent_epoch_id, lock=True)
        child = self._require_epoch(db, child_epoch_id, lock=True)
        if parent.owner_user_id != owner_id or child.owner_user_id != owner_id:
            raise LedgerConflict("Delegation epochs belong to different owners")

        if parent.run_id == child.run_id:
            raise LedgerConflict("Delegation must target a different Run")

        if parent.root_run_id != child.root_run_id:
            raise LedgerConflict("Delegation epochs have different root Runs")

        action = db.execute(
            select(Action).where(Action.id == parent_action_id)
        ).scalar_one_or_none()

        if action is None:
            raise LedgerNotFound(f"Parent Action not found: {parent_action_id}")

        if action.run_id != parent.run_id:
            raise LedgerConflict("Delegation Action does not belong to parent Run")

        if initiating_actor_id is not None:
            actor = self._require_actor(
                db,
                initiating_actor_id,
            )

            if actor.epoch_id != parent_epoch_id:
                raise LedgerConflict("Initiating actor belongs to another epoch")

        stmt = (
            select(RunDelegation)
            .where(RunDelegation.child_epoch_id == child_epoch_id)
            .with_for_update()
        )

        existing = db.execute(stmt).scalar_one_or_none()

        if existing is not None:
            same_relationship = (
                existing.parent_epoch_id == parent_epoch_id
                and existing.parent_action_id == parent_action_id
                and existing.initiating_actor_id == initiating_actor_id
            )

            if not same_relationship:
                raise LedgerConflict("Child epoch already has a different delegation")

            return existing

        delegation = RunDelegation(
            id=self._new_id(),
            parent_epoch_id=parent_epoch_id,
            child_epoch_id=child_epoch_id,
            parent_action_id=parent_action_id,
            initiating_actor_id=initiating_actor_id,
            created_at=self._now_seconds(),
        )

        db.add(delegation)
        db.flush()

        return delegation

    def append_event(
        self,
        db: Session,
        *,
        epoch_id: str,
        event_type: str,
        actor_id: Optional[str] = None,
        evidence_class: str = "structured",
        details: Optional[Mapping[str, Any]] = None,
        action_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        evidence_ref: Optional[str] = None,
        evidence_sha256: Optional[str] = None,
        occurred_at_us: Optional[int] = None,
    ) -> ExecutionEvent:
        """
        Append a structured event to an epoch's hash chain.

        The epoch row lock serializes event appends.

        Evidence is deliberately represented by an opaque reference
        and optional SHA-256, not by inline raw payloads.

        This hash chain detects unintended record alterations when
        compared with a trusted anchor. It is NOT a substitute for
        externally anchored, tamper-resistant evidence storage.
        """
        self._validate_identifier(epoch_id, "epoch_id")
        self._validate_code(
            event_type,
            "event_type",
            64,
        )

        if evidence_class not in self.RETENTION_DAYS:
            raise LedgerValidationError(f"Unsupported evidence class: {evidence_class}")

        clean_details = self._safe_details(details)

        if action_id is not None:
            self._validate_identifier(
                action_id,
                "action_id",
            )

        if tool_call_id is not None:
            self._validate_identifier(
                tool_call_id,
                "tool_call_id",
            )

        if evidence_ref is not None:
            self._validate_identifier(
                evidence_ref,
                "evidence_ref",
                255,
            )

            if "://" in evidence_ref or "?" in evidence_ref:
                raise LedgerValidationError(
                    "Evidence references must be opaque identifiers, "
                    "not URLs or signed access tokens"
                )

        if evidence_sha256 is not None:
            if (
                not isinstance(evidence_sha256, str)
                or re.fullmatch(
                    r"[0-9a-fA-F]{64}",
                    evidence_sha256,
                )
                is None
            ):
                raise LedgerValidationError(
                    "evidence_sha256 must be a SHA-256 hex digest"
                )

            evidence_sha256 = evidence_sha256.lower()

        epoch = self._require_epoch(
            db,
            epoch_id,
            lock=True,
        )

        if actor_id is not None:
            actor = self._require_actor(db, actor_id)

            if actor.epoch_id != epoch_id:
                raise LedgerConflict("Event actor belongs to another execution epoch")

        # Preserve the existing event ID ordering contract.
        # This assumes the ORM defines a monotonic event ID.
        # Verify against the released ExecutionEvent model.
        previous = db.execute(
            select(ExecutionEvent)
            .where(ExecutionEvent.epoch_id == epoch_id)
            .order_by(ExecutionEvent.id.desc())
            .limit(1)
            .with_for_update()
        ).scalar_one_or_none()

        previous_digest = previous.record_digest if previous is not None else None

        recorded_us = self._now_microseconds()

        if occurred_at_us is None:
            occurred_us = recorded_us
        else:
            if (
                not isinstance(occurred_at_us, int)
                or isinstance(occurred_at_us, bool)
                or occurred_at_us <= 0
            ):
                raise LedgerValidationError("occurred_at_us must be a positive integer")

            occurred_us = occurred_at_us

        retention_seconds = self.RETENTION_DAYS[evidence_class] * 86_400

        expires_at = self._now_seconds() + retention_seconds

        record = {
            "epoch_id": epoch.id,
            "actor_id": actor_id,
            "origin_run_id": epoch.run_id,
            "event_type": event_type,
            "schema_version": 1,
            "evidence_class": evidence_class,
            "occurred_at_us": occurred_us,
            "recorded_at_us": recorded_us,
            "evidence_expires_at": expires_at,
            "action_id_snapshot": action_id,
            "tool_call_id_snapshot": tool_call_id,
            "details": clean_details,
            "evidence_ref": evidence_ref,
            "evidence_sha256": evidence_sha256,
            "previous_digest": previous_digest,
        }

        canonical = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        event = ExecutionEvent(
            **record,
            record_digest=digest,
        )

        db.add(event)

        epoch.last_seen_at = self._now_seconds()

        db.flush()
        return event

    def complete_epoch(
        self,
        db: Session,
        *,
        epoch_id: str,
        terminal_status: str,
    ) -> RunExecutionEpoch:
        """
        Mark an epoch terminal without deleting any lineage.
        """
        if terminal_status not in self.TERMINAL_EPOCH_STATES:
            raise LedgerValidationError(f"Invalid terminal status: {terminal_status}")

        epoch = self._require_epoch(
            db,
            epoch_id,
            lock=True,
        )

        if epoch.state in self.TERMINAL_EPOCH_STATES:
            if epoch.state != terminal_status:
                raise LedgerConflict(
                    "Epoch has already terminated " "with another status"
                )

            return epoch

        now = self._now_seconds()

        epoch.state = terminal_status
        epoch.terminal_status = terminal_status
        epoch.ended_at = now
        epoch.last_seen_at = now

        db.flush()
        return epoch

    def create_retention_hold(
        self,
        db: Session,
        *,
        root_run_id: str,
        case_reference: str,
        reason_code: str,
        authorised_by: str,
    ) -> EvidenceRetentionHold:
        """
        Record an investigation hold.

        Caller must independently verify that authorised_by has the
        required authority. This method is not an authorization grant.

        Actual retention/purge enforcement belongs to the
        retention policy service.
        """
        self._validate_identifier(
            root_run_id,
            "root_run_id",
        )
        self._validate_identifier(
            case_reference,
            "case_reference",
            128,
        )
        self._validate_code(
            reason_code,
            "reason_code",
            48,
        )
        self._validate_identifier(
            authorised_by,
            "authorised_by",
        )

        self._require_run(
            db,
            root_run_id,
            lock=True,
        )

        hold = EvidenceRetentionHold(
            id=self._new_id(),
            root_run_id=root_run_id,
            case_reference=case_reference,
            reason_code=reason_code,
            authorised_by=authorised_by,
            created_at=self._now_seconds(),
        )

        db.add(hold)
        db.flush()

        return hold
