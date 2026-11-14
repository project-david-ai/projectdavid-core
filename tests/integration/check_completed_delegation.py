"""Read-only regression audit for a completed Deep Research execution."""

import sys

from sqlalchemy import select

from projectdavid_orm.projectdavid_orm.execution_ledger import (
    ExecutionActor,
    ExecutionEvent,
    RunDelegation,
    RunExecutionEpoch,
)
from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Action, Run


def state(value):
    return str(getattr(value, "value", value)).lower()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main(root_id):
    with SessionLocal() as db:
        root = db.get(Run, root_id)
        require(root is not None, "Root Run missing")
        require(state(root.status) == "completed", "Root Run incomplete")

        epochs = db.scalars(
            select(RunExecutionEpoch).where(RunExecutionEpoch.root_run_id == root_id)
        ).all()

        by_id = {epoch.id: epoch for epoch in epochs}
        roots = [e for e in epochs if e.run_id == root_id]

        require(len(roots) == 1, "Expected one root epoch")
        require(roots[0].state == "completed", "Root epoch incomplete")

        root_epoch = roots[0]

        links = db.scalars(
            select(RunDelegation).where(RunDelegation.parent_epoch_id == root_epoch.id)
        ).all()

        require(len(links) == 2, "Expected exactly two delegations")
        require(len(epochs) == 3, "Expected three execution epochs")

        root_actors = db.scalars(
            select(ExecutionActor).where(ExecutionActor.epoch_id == root_epoch.id)
        ).all()

        require(
            any(a.execution_role == "research_supervisor" for a in root_actors),
            "Research supervisor missing",
        )
        require(
            all(a.state == "retired" for a in root_actors),
            "Root execution actors not retired",
        )

        seen_children = set()
        seen_actions = set()

        for link in links:
            child_epoch = by_id.get(link.child_epoch_id)
            require(child_epoch is not None, "Linked child epoch missing")

            child = db.get(Run, child_epoch.run_id)
            action = db.get(Action, link.parent_action_id)

            require(child is not None, "Child Run missing")
            require(action is not None, "Parent Action missing")

            require(action.run_id == root_id, "Action belongs to wrong Run")
            require(
                state(action.status) == "completed",
                f"Parent Action {action.id} is {state(action.status)}",
            )
            require(
                state(child.status) == "completed",
                f"Child Run {child.id} is {state(child.status)}",
            )
            require(
                child_epoch.state == "completed",
                "Child execution epoch incomplete",
            )

            seen_children.add(child.id)
            seen_actions.add(action.id)

            workers = db.scalars(
                select(ExecutionActor).where(
                    ExecutionActor.epoch_id == child_epoch.id,
                    ExecutionActor.execution_role == "research_worker",
                )
            ).all()

            require(len(workers) == 1, "Expected one research-worker actor")
            worker = workers[0]

            require(worker.state == "retired", "Worker not retired")
            require(
                worker.assistant_id_snapshot == child.assistant_id,
                "Worker identity snapshot mismatch",
            )

            events = db.scalars(
                select(ExecutionEvent).where(ExecutionEvent.epoch_id == child_epoch.id)
            ).all()

            event_types = {e.event_type for e in events}

            require(
                "research_worker_assigned" in event_types,
                "Worker assignment event missing",
            )
            require(
                "research_worker_retired" in event_types,
                "Worker retirement event missing",
            )
            require(
                "run_completed" in event_types,
                "Worker completion event missing",
            )

            print(
                f"PASS child={child.id} "
                f"action={action.id} "
                f"status={state(action.status)}"
            )

        require(len(seen_children) == 2, "Duplicate child Run")
        require(len(seen_actions) == 2, "Duplicate parent Action")

        print(
            f"PASS root={root_id} " f"epochs={len(epochs)} " f"delegations={len(links)}"
        )
        print("EXECUTION LINEAGE REGRESSION PASSED")


if __name__ == "__main__":
    require(len(sys.argv) == 2, "Provide a root Run ID")
    main(sys.argv[1])
