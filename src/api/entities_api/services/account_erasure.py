"""Administrative erasure coordinator. Never infer completion from a missing user."""

import os
import time

from fastapi import HTTPException
from redis import Redis
from sqlalchemy import delete, or_, select, update

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import (
    ApiKey,
    Assistant,
    AuditLog,
    Base,
    Message,
    Run,
    Scratchpad,
    Thread,
    User,
)

from .erasure_state import (
    COMPLETE,
    PENDING,
    erasure_state,
    erasures,
    lock_user,
    subject_hash,
)
from .request_activity import request_activity
from .sandbox_drain import require_sandbox_drain

TERMINAL = {"completed", "failed", "cancelled", "expired"}


def _blocked():
    raise HTTPException(
        status_code=409,
        detail="Account deletion is pending resource reconciliation.",
    )


def _check_credential_boundary(db, tables, user_id: str) -> None:
    credentials = tables["credentials"]
    registrations = tables["mcp_server_registrations"]
    owned_credentials = select(credentials.c.id).where(
        credentials.c.owner_id == user_id
    )
    # A registration owned by another tenant must never be removed to
    # satisfy the credential foreign key. Keep the account pending instead.
    if db.execute(
        select(registrations.c.id)
        .where(
            registrations.c.credential_id.in_(owned_credentials),
            or_(
                registrations.c.owner_id != user_id,
                registrations.c.owner_id.is_(None),
            ),
        )
        .limit(1)
        .with_for_update()
    ).first():
        _blocked()


def _erase_owned_credentials(db, tables, user_id: str) -> None:
    # MySQL does not guarantee a safe ordering between sibling cascades.
    # Registration tool attachments cascade from the registration itself.
    registrations = tables["mcp_server_registrations"]
    credentials = tables["credentials"]
    db.execute(delete(registrations).where(registrations.c.owner_id == user_id))
    db.execute(delete(credentials).where(credentials.c.owner_id == user_id))


def begin_erasure(user_id: str) -> str:
    """
    Durably quarantine an account and revoke every API credential.

    No physical or relational resources are deleted here.

    Lock ordering is shared with credential issuance:
        1. User row.
        2. Erasure receipt.
        3. API credentials.
    """
    receipt_id = subject_hash(user_id)

    with SessionLocal() as db:
        user = lock_user(db, user_id)
        state = erasure_state(
            db,
            user_id,
            for_update=True,
        )

        if user is None:
            if state == COMPLETE:
                return COMPLETE

            raise HTTPException(
                status_code=404,
                detail="Erasure completion cannot be confirmed.",
            )

        if state not in (None, PENDING):
            _blocked()

        if state is None:
            db.execute(
                erasures.insert().values(
                    subject_hash=receipt_id,
                    state=PENDING,
                    created_at=int(time.time()),
                )
            )

        # Revoke ordinary and protected service credentials alike.
        db.execute(
            update(ApiKey).where(ApiKey.user_id == user_id).values(is_active=False)
        )

        db.commit()

    return PENDING


def erase_user(storage, user_id: str) -> None:
    """
    Attempt account erasure after durable quarantine.

    Failed reconciliation preserves the pending receipt and database
    locators so that the operation can be retried.

    The quarantine boundary does not independently establish that
    previously admitted requests or detached workers have stopped.
    """
    receipt_id = subject_hash(user_id)

    if begin_erasure(user_id) == COMPLETE:
        return

    # Registration precedes the fresh authentication receipt read.
    # Once quarantine commits, an already-admitted HTTP request is
    # still counted; a newly registered request cannot pass admission.
    # This is a necessary local check, not proof of detached-worker drain.
    if request_activity.active_count(user_id):
        raise HTTPException(
            status_code=409,
            detail="Account deletion is pending active request reconciliation.",
        )

    with SessionLocal() as db:
        # Canonical lock order: user row, then erasure receipt.
        user = lock_user(db, user_id)
        state = erasure_state(
            db,
            user_id,
            for_update=True,
        )

        if user is None and state == COMPLETE:
            return

        if user is None or state != PENDING:
            _blocked()

        tables = Base.metadata.tables

        # -------------------------------------------------------------
        # Execution reconciliation
        # -------------------------------------------------------------

        runs = list(db.scalars(select(Run).where(Run.user_id == user_id)))

        run_ids = [run.id for run in runs]

        if any(
            str(getattr(run.status, "value", run.status)) not in TERMINAL
            for run in runs
        ):
            # Status-only cancellation is not proof that workers stopped.
            _blocked()

        # These resource classes have external runtimes or artifacts
        # without a verified account-erasure acknowledgement.
        # Preserve their locators for reconciliation.
        for name in (
            "sandboxes",
            "datasets",
            "training_jobs",
            "fine_tuned_models",
        ):
            table = tables[name]

            if db.execute(
                select(table.c.id).where(table.c.user_id == user_id).limit(1)
            ).first():
                _blocked()

        # -------------------------------------------------------------
        # Execution ledger and retention holds
        # -------------------------------------------------------------

        epochs = tables["run_execution_epochs"]
        actors = tables["execution_actors"]
        events = tables["execution_events"]
        delegations = tables["run_delegations"]
        holds = tables["evidence_retention_holds"]

        epoch_rows = (
            db.execute(
                select(epochs).where(
                    or_(
                        epochs.c.owner_user_id == user_id,
                        epochs.c.run_id.in_(run_ids),
                        epochs.c.root_run_id.in_(run_ids),
                    )
                )
            )
            .mappings()
            .all()
        )

        if any(
            row["owner_user_id"] != user_id or row["state"] not in TERMINAL
            for row in epoch_rows
        ):
            _blocked()

        epoch_ids = [row["id"] for row in epoch_rows]

        # Every associated execution actor must be retired.
        if db.execute(
            select(actors.c.id)
            .where(
                actors.c.epoch_id.in_(epoch_ids),
                actors.c.state != "retired",
            )
            .limit(1)
        ).first():
            _blocked()

        # Never bypass an investigation hold, including one
        # awaiting review.
        if db.execute(
            select(holds.c.id)
            .where(
                holds.c.root_run_id.in_(run_ids),
                holds.c.released_at.is_(None),
            )
            .limit(1)
        ).first():
            _blocked()

        links = (
            db.execute(
                select(delegations).where(
                    or_(
                        delegations.c.parent_epoch_id.in_(epoch_ids),
                        delegations.c.child_epoch_id.in_(epoch_ids),
                    )
                )
            )
            .mappings()
            .all()
        )

        # Do not delete a delegation spanning an execution
        # outside this account's erasure boundary.
        if any(
            row["parent_epoch_id"] not in epoch_ids
            or row["child_epoch_id"] not in epoch_ids
            for row in links
        ):
            _blocked()

        # -------------------------------------------------------------
        # Resource ownership and sharing
        # -------------------------------------------------------------

        owned_threads = list(
            db.scalars(select(Thread).where(Thread.owner_id == user_id))
        )

        exclusive_threads = [
            thread.id
            for thread in owned_threads
            if all(participant.id == user_id for participant in thread.participants)
        ]

        owned_assistants = list(
            db.scalars(select(Assistant).where(Assistant.owner_id == user_id))
        )

        exclusive_assistants = [
            assistant.id
            for assistant in owned_assistants
            if all(associated_user.id == user_id for associated_user in assistant.users)
        ]

        # Block deletion when another user's execution references
        # a resource classified as exclusively owned.
        if db.execute(
            select(Run.id)
            .where(
                Run.user_id != user_id,
                or_(
                    Run.thread_id.in_(exclusive_threads),
                    Run.assistant_id.in_(exclusive_assistants),
                ),
            )
            .limit(1)
        ).first():
            _blocked()

        # -------------------------------------------------------------
        # Snapshot resource locators
        # -------------------------------------------------------------

        affected_threads = set(exclusive_threads)

        affected_threads.update(
            db.scalars(
                select(Message.thread_id).where(
                    or_(
                        Message.sender_id == user_id,
                        Message.run_id.in_(run_ids),
                    )
                )
            )
        )

        scratchpads = list(
            db.scalars(select(Scratchpad).where(Scratchpad.owner_id == user_id))
        )

        # -------------------------------------------------------------
        # Physical storage reconciliation
        # -------------------------------------------------------------

        # Physical cleanup must precede removal of durable locators.
        # Exceptions leave the pending receipt and relational data intact.
        _check_credential_boundary(db, tables, user_id)
        require_sandbox_drain(user_id)
        storage._erase_physical_files(db, user_id)
        storage._erase_vector_store_collections(db, user_id)

        # -------------------------------------------------------------
        # Redis cache reconciliation
        # -------------------------------------------------------------

        cache = Redis.from_url(
            os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0",
            ),
            socket_timeout=10,
            socket_connect_timeout=5,
        )

        try:
            keys = [
                f"assistant:{identifier}:config" for identifier in exclusive_assistants
            ]

            keys += [f"thread:{identifier}:history" for identifier in affected_threads]

            keys += [
                f"scratchpad:{identifier}:notebook" for identifier in exclusive_threads
            ]

            keys += [
                f"scratchpad:{user_id}:{scratchpad.id}:{suffix}"
                for scratchpad in scratchpads
                for suffix in ("content", "entries")
            ]

            # Escape Redis glob metacharacters even though generated
            # user identifiers normally contain none.
            escaped = "".join(
                "\\" + character if character in "*?[]\\" else character
                for character in user_id
            )

            keys += list(
                cache.scan_iter(
                    match=f"net_eng:usr:{escaped}:inv:*",
                    count=100,
                )
            )

            for offset in range(0, len(keys), 100):
                cache.delete(*keys[offset : offset + 100])

        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Cache erasure is incomplete; retry required.",
            ) from None

        finally:
            cache.close()

        # -------------------------------------------------------------
        # Relational erasure
        # -------------------------------------------------------------

        db.execute(
            delete(events).where(
                or_(
                    events.c.epoch_id.in_(epoch_ids),
                    events.c.origin_run_id.in_(run_ids),
                )
            )
        )

        db.execute(
            delete(delegations).where(delegations.c.parent_epoch_id.in_(epoch_ids))
        )

        db.execute(delete(actors).where(actors.c.epoch_id.in_(epoch_ids)))

        db.execute(delete(epochs).where(epochs.c.id.in_(epoch_ids)))

        db.execute(delete(holds).where(holds.c.root_run_id.in_(run_ids)))

        db.execute(
            delete(Message).where(
                or_(
                    Message.thread_id.in_(exclusive_threads),
                    Message.sender_id == user_id,
                    Message.run_id.in_(run_ids),
                )
            )
        )

        db.execute(delete(Scratchpad).where(Scratchpad.owner_id == user_id))

        db.execute(delete(Thread).where(Thread.id.in_(exclusive_threads)))

        db.execute(delete(Assistant).where(Assistant.id.in_(exclusive_assistants)))

        # -------------------------------------------------------------
        # Audit data minimisation
        # -------------------------------------------------------------

        # Preserve action/type/time evidence while removing direct
        # identifiers and arbitrary historical payloads.
        # This is minimisation, not guaranteed anonymity.
        audit_ids = [
            user_id,
            *exclusive_threads,
            *exclusive_assistants,
            *run_ids,
        ]

        db.execute(
            update(AuditLog)
            .where(
                or_(
                    AuditLog.user_id == user_id,
                    AuditLog.entity_id.in_(audit_ids),
                )
            )
            .values(
                user_id=None,
                entity_id="erased",
                ip_address=None,
                details=None,
            )
        )

        # -------------------------------------------------------------
        # Account deletion and durable completion receipt
        # -------------------------------------------------------------

        _erase_owned_credentials(db, tables, user_id)

        # SQL DELETE deliberately avoids ORM relationship nullification.
        db.execute(delete(User).where(User.id == user_id))

        # The receipt and relational deletion commit atomically.
        db.execute(
            update(erasures)
            .where(erasures.c.subject_hash == receipt_id)
            .values(
                state=COMPLETE,
                completed_at=int(time.time()),
            )
        )

        db.commit()
