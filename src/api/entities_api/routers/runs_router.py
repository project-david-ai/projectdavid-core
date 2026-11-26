# src/api/entities_api/routers/runs.py
import asyncio
import json
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.execution_ledger import (
    ExecutionActor,
    ExecutionEvent,
    RunDelegation,
    RunExecutionEpoch,
)
from pydantic import ValidationError
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, aliased
from sse_starlette.sse import EventSourceResponse
from starlette import status

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.models.models import Run
from src.api.entities_api.models.models import User as UserModel
from src.api.entities_api.services.actions_service import ActionService
from src.api.entities_api.services.execution_ledger_service import (
    ExecutionLedgerService,
)
from src.api.entities_api.services.runs_service import RunService

ent_validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()
router = APIRouter()


@router.post("/runs", response_model=ValidationInterface.Run)
def create_run(
    run: ValidationInterface.RunCreate,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id
    logging_utility.info("[%s] Creating run for thread %s", user_id, run.thread_id)
    run_service = RunService()
    try:
        new_run = run_service.create_run(
            run,
            user_id=user_id,
            is_protected_key=bool(auth_key.is_protected),
        )
        logging_utility.info("Run created successfully: %s", new_run.id)
        return new_run
    except HTTPException as e:
        logging_utility.error("HTTP error during run creation: %s", str(e))
        raise
    except Exception as e:
        logging_utility.error("Unexpected error during run creation: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get("/runs/{run_id}", response_model=ValidationInterface.RunReadDetailed)
def retrieve_run(
    run_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Retrieving run ID: {run_id}")
    run_service = RunService()
    try:
        run = run_service.retrieve_run(run_id, user_id=auth_key.user_id)  # ← FIXED
        logging_utility.info(f"Run retrieved successfully: {run_id}")
        return run
    except HTTPException as e:
        logging_utility.error(f"HTTP error retrieving run {run_id}: {str(e)}")
        raise e
    except Exception as e:
        logging_utility.error(f"Unexpected error retrieving run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.put("/runs/{run_id}/status", response_model=ValidationInterface.Run)
def update_run_status(
    run_id: str,
    status_update: ValidationInterface.RunStatusUpdate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Updating status of run {run_id} → {status_update.status}"
    )
    # Admin gate — ownership not checked; admin privilege is the guard here.
    requesting_admin = (
        db.query(UserModel).filter(UserModel.id == auth_key.user_id).first()
    )
    if not requesting_admin or not requesting_admin.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required to update run status.",
        )

    run_service = RunService()
    try:
        updated_run = run_service.update_run_status(run_id, status_update.status)
        logging_utility.info(f"Run status updated: {run_id}")
        return updated_run
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e.errors()}")
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.patch("/runs/{run_id}/fields", response_model=ValidationInterface.Run)
def update_run_fields(
    run_id: str,
    payload: Dict[str, Any] = Body(...),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    """
    Targeted mid-run lifecycle field update.

    Accepts any subset of the MUTABLE_RUN_FIELDS whitelist:
        started_at, completed_at, failed_at, last_error,
        incomplete_details, current_turn, usage, meta_data

    All other keys in the payload are silently ignored by the service layer.
    No admin privileges required — the orchestrator calls this internally
    on behalf of the run owner.
    """
    logging_utility.info(
        "[%s] PATCH fields for run %s: %s",
        auth_key.user_id,
        run_id,
        list(payload.keys()),
    )
    svc = RunService()
    try:
        return svc.update_run_fields(  # ← FIXED: user_id now forwarded
            run_id, user_id=auth_key.user_id, **payload
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logging_utility.error(
            "Unexpected error patching run %s fields: %s", run_id, str(e)
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.put("/runs/{run_id}/metadata", response_model=ValidationInterface.Run)
def update_run_metadata(
    run_id: str,
    payload: Dict[str, Any] = Body(...),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info("[%s] Updating metadata for run %s", auth_key.user_id, run_id)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    metadata = (
        payload["metadata"]
        if "metadata" in payload and isinstance(payload["metadata"], dict)
        else payload
    )
    svc = RunService()
    # user_id already forwarded in original — no change needed here.
    return svc.update_run(run_id, metadata, user_id=auth_key.user_id)


@router.post("/runs/{run_id}/cancel", response_model=ValidationInterface.Run)
def cancel_run(
    run_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Cancelling run {run_id}")
    run_service = RunService()
    try:
        cancelled_run = run_service.cancel_run(  # ← FIXED: user_id now forwarded
            run_id, user_id=auth_key.user_id
        )
        logging_utility.info(f"Run cancelled successfully: {run_id}")
        return cancelled_run
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get(
    "/runs/{run_id}/events",
    summary="Stream run-lifecycle events (SSE)",
    response_class=EventSourceResponse,
)
async def stream_run_events(
    request: Request,
    run_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    run_svc = RunService()
    action_svc = ActionService()
    owner_id = auth_key.user_id

    # ── Ownership verified once at connection time, before the stream opens. ──
    # Raises 403 immediately if the caller doesn't own this run.
    # The internal poll loop below omits user_id — no repeated auth overhead.
    try:
        run_svc.retrieve_run(run_id, user_id=owner_id)
    except HTTPException as e:
        raise e

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            # Internal poll — user_id omitted intentionally.
            run = run_svc.retrieve_run(run_id)
            if not run:
                yield {"event": "error", "data": '{"msg":"run not found"}'}
                break
            if run.status == StatusEnum.pending_action:
                pending = action_svc.get_pending_actions(run_id, user_id=owner_id)
                if pending:
                    for act in pending:
                        data = act.dict() if hasattr(act, "dict") else act
                        yield {"event": "action_required", "data": json.dumps(data)}
                    break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/runs", response_model=ValidationInterface.RunListResponse)
def list_runs(
    limit: int = Query(20, ge=1, le=100),
    order: Literal["asc", "desc"] = Query("asc"),
    thread_id: Optional[str] = Query(None),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id
    logging_utility.info(
        "[%s] Listing runs (limit=%s, order=%s, thread_id=%s)",
        user_id,
        limit,
        order,
        thread_id,
    )
    svc = RunService()
    try:
        runs, has_more = svc.list_runs(
            user_id=user_id, limit=limit, order=order, thread_id=thread_id
        )
        return {
            "object": "list",
            "data": runs,
            "first_id": runs[0].id if runs else None,
            "last_id": runs[-1].id if runs else None,
            "has_more": has_more,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logging_utility.error(
            "list_runs 500 for user %s:\n%s",
            user_id,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/assistants/{assistant_id}/runs",
    response_model=ValidationInterface.RunListResponse,
)
def list_runs_for_assistant(
    assistant_id: str,
    limit: int = Query(20, ge=1, le=100),
    order: Literal["asc", "desc"] = Query("desc"),
    after: Optional[str] = Query(None, min_length=1, max_length=64),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    try:
        runs, has_more = RunService().list_runs_by_assistant_id(
            assistant_id,
            user_id=auth_key.user_id,
            limit=limit,
            order=order,
            after=after,
        )
        return {
            "object": "list",
            "data": runs,
            "first_id": runs[0].id if runs else None,
            "last_id": runs[-1].id if runs else None,
            "has_more": has_more,
        }
    except HTTPException:
        raise
    except Exception:
        logging_utility.error("Failed to list runs for assistant %s", assistant_id)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        ) from None


@router.get(
    "/threads/{thread_id}/runs",
    response_model=ValidationInterface.RunListResponse,
)
def list_runs_for_thread(
    thread_id: str,
    limit: int = Query(20, ge=1, le=100),
    order: Literal["asc", "desc"] = Query("asc"),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    user_id = auth_key.user_id
    logging_utility.info(
        "[%s] Listing runs for thread %s (limit=%s, order=%s)",
        user_id,
        thread_id,
        limit,
        order,
    )
    svc = RunService()
    try:
        runs, has_more = svc.list_runs(
            user_id=user_id, limit=limit, order=order, thread_id=thread_id
        )
        return {
            "object": "list",
            "data": runs,
            "first_id": runs[0].id if runs else None,
            "last_id": runs[-1].id if runs else None,
            "has_more": has_more,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


# PROJECTDAVID_EXECUTION_LEDGER_READ_PATCH


def _ledger_owned_root(db, run_id, owner_id):
    owned = db.execute(
        select(Run.id).where(
            Run.id == run_id,
            Run.user_id == owner_id,
        )
    ).scalar_one_or_none()

    if owned is None:
        raise HTTPException(404, "Run not found.")

    root = (
        db.execute(
            select(RunExecutionEpoch.root_run_id)
            .where(
                RunExecutionEpoch.run_id == run_id,
                RunExecutionEpoch.owner_user_id == owner_id,
            )
            .order_by(RunExecutionEpoch.epoch_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        or run_id
    )

    root_owned = db.execute(
        select(Run.id).where(
            Run.id == root,
            Run.user_id == owner_id,
        )
    ).scalar_one_or_none()

    if root_owned is None:
        raise HTTPException(404, "Run not found.")

    return root


def _ledger_page(data, root, cursor):
    return {
        "object": "list",
        "data": data,
        "has_more": cursor is not None,
        "next_cursor": cursor,
        "root_run_id": root,
    }


@router.get("/runs/{run_id}/execution")
def list_run_execution(
    run_id: str,
    limit: int = Query(25, ge=1, le=100),
    after: Optional[str] = Query(None, min_length=1, max_length=64),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    root = _ledger_owned_root(db, run_id, auth_key.user_id)

    scope = select(RunExecutionEpoch).where(
        RunExecutionEpoch.root_run_id == root,
        RunExecutionEpoch.owner_user_id == auth_key.user_id,
    )

    if after is not None:
        anchor = db.execute(
            scope.where(RunExecutionEpoch.id == after)
        ).scalar_one_or_none()

        if anchor is None:
            raise HTTPException(422, "Invalid execution cursor.")

        scope = scope.where(
            or_(
                RunExecutionEpoch.created_at > anchor.created_at,
                and_(
                    RunExecutionEpoch.created_at == anchor.created_at,
                    RunExecutionEpoch.id > anchor.id,
                ),
            )
        )

    epochs = (
        db.execute(
            scope.order_by(
                RunExecutionEpoch.created_at,
                RunExecutionEpoch.id,
            ).limit(limit + 1)
        )
        .scalars()
        .all()
    )

    more = len(epochs) > limit
    epochs = epochs[:limit]

    actors_by_epoch = {epoch.id: [] for epoch in epochs}

    if actors_by_epoch:
        actors = (
            db.execute(
                select(ExecutionActor)
                .where(ExecutionActor.epoch_id.in_(list(actors_by_epoch)))
                .order_by(ExecutionActor.reserved_at, ExecutionActor.id)
            )
            .scalars()
            .all()
        )

        for actor in actors:
            actors_by_epoch[actor.epoch_id].append(
                {
                    "id": actor.id,
                    "epoch_id": actor.epoch_id,
                    "slot": actor.slot,
                    "execution_role": actor.execution_role,
                    "assistant_id_snapshot": actor.assistant_id_snapshot,
                    "model_snapshot": actor.model_snapshot,
                    "ephemeral": actor.ephemeral,
                    "state": actor.state,
                    "reserved_at": actor.reserved_at,
                    "activated_at": actor.activated_at,
                    "retired_at": actor.retired_at,
                    "last_seen_at": actor.last_seen_at,
                }
            )

    data = [
        {
            "id": epoch.id,
            "run_id": epoch.run_id,
            "root_run_id": epoch.root_run_id,
            "epoch_number": epoch.epoch_number,
            "state": epoch.state,
            "created_at": epoch.created_at,
            "started_at": epoch.started_at,
            "last_seen_at": epoch.last_seen_at,
            "ended_at": epoch.ended_at,
            "terminal_status": epoch.terminal_status,
            "actors": actors_by_epoch[epoch.id],
        }
        for epoch in epochs
    ]

    return _ledger_page(
        data,
        root,
        epochs[-1].id if more else None,
    )


@router.get("/runs/{run_id}/delegations")
def list_run_delegations(
    run_id: str,
    limit: int = Query(25, ge=1, le=100),
    after: Optional[str] = Query(None, min_length=1, max_length=64),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    root = _ledger_owned_root(db, run_id, auth_key.user_id)

    parent = aliased(RunExecutionEpoch)
    child = aliased(RunExecutionEpoch)
    parent_run = aliased(Run)
    child_run = aliased(Run)

    scope = (
        select(RunDelegation, parent.run_id, child.run_id)
        .join(parent, RunDelegation.parent_epoch_id == parent.id)
        .join(child, RunDelegation.child_epoch_id == child.id)
        .join(parent_run, parent.run_id == parent_run.id)
        .join(child_run, child.run_id == child_run.id)
        .where(
            parent.root_run_id == root,
            child.root_run_id == root,
            parent.owner_user_id == auth_key.user_id,
            child.owner_user_id == auth_key.user_id,
            parent_run.user_id == auth_key.user_id,
            child_run.user_id == auth_key.user_id,
        )
    )

    if after is not None:
        anchor = db.execute(scope.where(RunDelegation.id == after)).one_or_none()

        if anchor is None:
            raise HTTPException(422, "Invalid delegation cursor.")

        record = anchor[0]
        scope = scope.where(
            or_(
                RunDelegation.created_at > record.created_at,
                and_(
                    RunDelegation.created_at == record.created_at,
                    RunDelegation.id > record.id,
                ),
            )
        )

    rows = db.execute(
        scope.order_by(
            RunDelegation.created_at,
            RunDelegation.id,
        ).limit(limit + 1)
    ).all()

    more = len(rows) > limit
    rows = rows[:limit]

    data = [
        {
            "id": item.id,
            "root_run_id": root,
            "parent_epoch_id": item.parent_epoch_id,
            "parent_run_id": parent_id,
            "parent_action_id": item.parent_action_id,
            "child_epoch_id": item.child_epoch_id,
            "child_run_id": child_id,
            "initiating_actor_id": item.initiating_actor_id,
            "created_at": item.created_at,
        }
        for item, parent_id, child_id in rows
    ]

    return _ledger_page(
        data,
        root,
        rows[-1][0].id if more else None,
    )


@router.get("/runs/{run_id}/execution-events")
def list_run_execution_events(
    run_id: str,
    limit: int = Query(25, ge=1, le=100),
    after: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    root = _ledger_owned_root(db, run_id, auth_key.user_id)

    scope = (
        select(ExecutionEvent)
        .join(
            RunExecutionEpoch,
            ExecutionEvent.epoch_id == RunExecutionEpoch.id,
        )
        .where(
            RunExecutionEpoch.root_run_id == root,
            RunExecutionEpoch.owner_user_id == auth_key.user_id,
            ExecutionEvent.origin_run_id == RunExecutionEpoch.run_id,
        )
    )

    if after is not None:
        scope = scope.where(ExecutionEvent.id > after)

    events = (
        db.execute(scope.order_by(ExecutionEvent.id).limit(limit + 1)).scalars().all()
    )

    more = len(events) > limit
    events = events[:limit]

    allowed = ExecutionLedgerService.SAFE_EVENT_DETAIL_KEYS

    data = []

    for event in events:
        details = event.details if isinstance(event.details, dict) else {}

        safe_details = {
            key: value
            for key, value in details.items()
            if key in allowed
            and isinstance(value, (str, int, float, bool))
            and (
                not isinstance(value, str)
                or (len(value) <= 128 and value.isprintable())
            )
        }

        data.append(
            {
                "id": event.id,
                "epoch_id": event.epoch_id,
                "actor_id": event.actor_id,
                "origin_run_id": event.origin_run_id,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "evidence_class": event.evidence_class,
                "occurred_at_us": event.occurred_at_us,
                "recorded_at_us": event.recorded_at_us,
                "evidence_expires_at": event.evidence_expires_at,
                "action_id_snapshot": event.action_id_snapshot,
                "tool_call_id_snapshot": event.tool_call_id_snapshot,
                "details": safe_details,
                "evidence_ref": None,
                "evidence_sha256": event.evidence_sha256,
                "previous_digest": event.previous_digest,
                "record_digest": event.record_digest,
            }
        )

    return _ledger_page(
        data,
        root,
        events[-1].id if more else None,
    )
