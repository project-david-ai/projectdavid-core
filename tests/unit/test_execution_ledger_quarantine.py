"""New execution lineage must not bypass durable account quarantine."""

import pytest
from projectdavid_common.validation import StatusEnum
from projectdavid_orm.projectdavid_orm.execution_ledger import (
    ExecutionActor,
    RunDelegation,
    RunExecutionEpoch,
)
from sqlalchemy import select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import Action, ApiKey, Run, User
from src.api.entities_api.services import account_erasure
from src.api.entities_api.services.erasure_state import (
    COMPLETE,
    PENDING,
    erasure_state,
    erasures,
    subject_hash,
)
from src.api.entities_api.services.execution_ledger_service import (
    ExecutionLedgerService,
    LedgerConflict,
)


def _run(run_id, owner_id):
    return Run(
        id=run_id,
        user_id=owner_id,
        assistant_id="assistant_test",
        thread_id="thread_test",
        object="thread.run",
        status=StatusEnum.queued,
    )


@pytest.fixture
def ledger_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        for model in (
            User,
            ApiKey,
            Run,
            Action,
            RunExecutionEpoch,
            ExecutionActor,
            RunDelegation,
        ):
            model.__table__.create(engine)
        erasures.create(engine)
        factory = sessionmaker(engine)
        monkeypatch.setattr(account_erasure, "SessionLocal", factory)
        ledger = ExecutionLedgerService()
        with factory.begin() as db:
            db.add_all([User(id="owner"), User(id="unrelated")])
            db.flush()
            db.add_all(
                [
                    _run("root", "owner"),
                    _run("child", "owner"),
                    _run("fresh", "owner"),
                    _run("terminal", "owner"),
                    _run("other_root", "unrelated"),
                    _run("other_child", "unrelated"),
                    _run("ownerless", None),
                ]
            )
            db.flush()
            db.add_all(
                [
                    Action(id="action_root", run_id="root"),
                    Action(id="action_other", run_id="other_root"),
                ]
            )
            db.flush()
            parent = ledger.reserve_epoch(db, run_id="root")
            child = ledger.reserve_epoch(db, run_id="child", root_run_id="root")
            actor = ledger.reserve_actor(
                db,
                epoch_id=parent.id,
                slot="supervisor",
                execution_role="research_supervisor",
            )
            terminal = ledger.reserve_epoch(db, run_id="terminal")
            ledger.complete_epoch(db, epoch_id=terminal.id, terminal_status="completed")
            refs = {"parent": parent.id, "child": child.id, "actor": actor.id}
        yield factory, refs
    finally:
        engine.dispose()


def _snapshot(factory):
    with factory() as db:
        return [
            [
                dict(row)
                for row in db.execute(select(table).order_by(table.c.id)).mappings()
            ]
            for table in (
                RunExecutionEpoch.__table__,
                ExecutionActor.__table__,
                RunDelegation.__table__,
            )
        ]


@pytest.mark.parametrize("state", [PENDING, COMPLETE])
@pytest.mark.parametrize(
    "operation",
    [
        "new_epoch",
        "existing_epoch",
        "new_attempt",
        "reserve_actor",
        "activate_actor",
        "link_delegation",
    ],
)
def test_erasure_receipt_blocks_ledger_admission(ledger_db, state, operation):
    factory, refs = ledger_db
    if state == PENDING:
        assert account_erasure.begin_erasure("owner") == PENDING
    else:
        # Defensive case: an unexpected residual User coexists with a receipt.
        with factory.begin() as db:
            db.execute(
                erasures.insert().values(
                    subject_hash=subject_hash("owner"),
                    state=COMPLETE,
                    created_at=100,
                    completed_at=200,
                )
            )
    before = _snapshot(factory)
    ledger = ExecutionLedgerService()
    with pytest.raises(LedgerConflict, match="owner is unavailable for admission"):
        with factory.begin() as db:
            if operation == "new_epoch":
                ledger.reserve_epoch(db, run_id="fresh")
            elif operation == "existing_epoch":
                ledger.reserve_epoch(db, run_id="root")
            elif operation == "new_attempt":
                ledger.reserve_epoch(db, run_id="terminal", new_attempt=True)
            elif operation == "reserve_actor":
                ledger.reserve_actor(
                    db, epoch_id=refs["parent"], slot="worker", execution_role="worker"
                )
            elif operation == "activate_actor":
                ledger.activate_actor(
                    db, actor_id=refs["actor"], assistant_id="ephemeral_test"
                )
            else:
                ledger.link_delegation(
                    db,
                    parent_epoch_id=refs["parent"],
                    child_epoch_id=refs["child"],
                    parent_action_id="action_root",
                )
    assert _snapshot(factory) == before
    with factory() as db:
        assert erasure_state(db, "owner") == state


def test_pending_account_can_retire_actor_and_complete_epoch(ledger_db):
    factory, refs = ledger_db
    assert account_erasure.begin_erasure("owner") == PENDING
    ledger = ExecutionLedgerService()
    with factory.begin() as db:
        ledger.retire_actor(db, actor_id=refs["actor"])
        ledger.complete_epoch(db, epoch_id=refs["parent"], terminal_status="cancelled")
    with factory() as db:
        assert db.get(ExecutionActor, refs["actor"]).state == "retired"
        assert db.get(RunExecutionEpoch, refs["parent"]).state == "cancelled"
        assert erasure_state(db, "owner") == PENDING


def test_unrelated_account_can_admit_execution(ledger_db):
    factory, refs = ledger_db
    assert account_erasure.begin_erasure("owner") == PENDING
    ledger = ExecutionLedgerService()
    with factory.begin() as db:
        parent = ledger.reserve_epoch(db, run_id="other_root")
        child = ledger.reserve_epoch(db, run_id="other_child", root_run_id="other_root")
        actor = ledger.reserve_actor(
            db,
            epoch_id=parent.id,
            slot="supervisor",
            execution_role="research_supervisor",
        )
        ledger.activate_actor(db, actor_id=actor.id, assistant_id="other_assistant")
        edge = ledger.link_delegation(
            db,
            parent_epoch_id=parent.id,
            child_epoch_id=child.id,
            parent_action_id="action_other",
            initiating_actor_id=actor.id,
        )
        parent_id, actor_id, edge_id = parent.id, actor.id, edge.id
    with factory() as db:
        assert db.get(RunExecutionEpoch, parent_id).state == "running"
        assert db.get(ExecutionActor, actor_id).state == "active"
        assert db.get(RunDelegation, edge_id).parent_epoch_id == parent_id
        assert db.get(ExecutionActor, refs["actor"]).state == "reserved"
        assert erasure_state(db, "owner") == PENDING
        assert erasure_state(db, "unrelated") is None


def test_ownerless_run_cannot_reserve_epoch(ledger_db):
    factory, _ = ledger_db
    with pytest.raises(LedgerConflict, match="Execution has no owner"):
        with factory.begin() as db:
            ExecutionLedgerService().reserve_epoch(db, run_id="ownerless")
    with factory() as db:
        assert db.query(RunExecutionEpoch).filter_by(run_id="ownerless").count() == 0
