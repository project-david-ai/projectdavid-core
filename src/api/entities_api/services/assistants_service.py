# src/api/entities_api/services/assistants_service.py

import base64
import binascii
import json
import re
import time
from typing import Any, List, Literal

from fastapi import HTTPException
from projectdavid import Entity
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_orm.projectdavid_orm.models import AssistantMcpTool
from pydantic import BaseModel, Field
from sqlalchemy import and_, or_, select, union_all

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Assistant, User, VectorStore
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.mcp_tool_config import (
    merge_user_tools_preserving_managed,
)
from src.api.entities_api.utilities.cache_utils import get_sync_invalidator

logging_utility = LoggingUtility()
validator = ValidationInterface()

Mode = Literal["standard", "autonomous"]
MAX_CURSOR = 512


class SummaryFilters(BaseModel):
    limit: int = Field(default=50, ge=1, le=100)
    cursor: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_CURSOR,
    )
    assistant_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )
    name_prefix: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )
    mode: Mode | None = None


class AssistantSummary(BaseModel):
    id: str
    name: str
    description: str | None
    model: str | None
    created_at: int
    agent_mode: bool
    web_access: bool
    deep_research: bool
    engineer: bool
    decision_telemetry: bool


class SummaryPage(BaseModel):
    items: list[AssistantSummary]
    next_cursor: str | None
    has_more: bool


def encode_cursor(created_at: int, assistant_id: str) -> str:
    payload = json.dumps(
        [1, created_at, assistant_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, str]:
    try:
        if len(cursor) > MAX_CURSOR or not re.fullmatch(
            r"[A-Za-z0-9_-]+",
            cursor,
        ):
            raise ValueError

        data = json.loads(
            base64.b64decode(
                cursor + "=" * (-len(cursor) % 4),
                altchars=b"-_",
                validate=True,
            )
        )

        if not isinstance(data, list) or len(data) != 3:
            raise ValueError

        version, ts, ident = data

        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != 1
            or not isinstance(ts, int)
            or isinstance(ts, bool)
            or not 0 <= ts <= 2**31 - 1
        ):
            raise ValueError

        if not isinstance(ident, str) or not 1 <= len(ident) <= 64:
            raise ValueError

        if encode_cursor(ts, ident) != cursor:
            raise ValueError

        return ts, ident

    except (ValueError, UnicodeError, binascii.Error, RecursionError):
        raise HTTPException(422, "Invalid assistant cursor") from None


def summary_query(user_id: str, filters: SummaryFilters):
    """Build a bounded, projected query without loading ORM relationships."""
    a = Assistant.__table__
    assoc = Assistant.users.property.secondary

    columns = [a.c[name] for name in AssistantSummary.model_fields]
    conditions = [a.c.deleted_at.is_(None)]

    if filters.assistant_id is not None:
        conditions.append(a.c.id == filters.assistant_id)

    if filters.name_prefix is not None:
        literal = (
            filters.name_prefix.replace("/", "//").replace("%", "/%").replace("_", "/_")
        )
        conditions.append(a.c.name.like(literal + "%", escape="/"))

    if filters.model is not None:
        conditions.append(a.c.model == filters.model)

    if filters.mode is not None:
        conditions.append(a.c.agent_mode == (filters.mode == "autonomous"))

    if filters.cursor:
        ts, ident = decode_cursor(filters.cursor)
        conditions.append(
            or_(
                a.c.created_at < ts,
                and_(
                    a.c.created_at == ts,
                    a.c.id < ident,
                ),
            )
        )

    order = (a.c.created_at.desc(), a.c.id.desc())
    cap = filters.limit + 1

    # Separate disjoint ownership branches so canonical reads can use
    # the owner/order index and legacy reads can start at associations.
    owned = (
        select(*columns)
        .where(a.c.owner_id == user_id, *conditions)
        .order_by(*order)
        .limit(cap)
        .subquery()
    )

    legacy = (
        select(*columns)
        .select_from(
            assoc.join(
                a,
                a.c.id == assoc.c.assistant_id,
            )
        )
        .where(
            assoc.c.user_id == user_id,
            a.c.owner_id.is_(None),
            *conditions,
        )
        .order_by(*order)
        .limit(cap)
        .subquery()
    )

    merged = union_all(select(owned), select(legacy)).subquery()

    return (
        select(merged)
        .order_by(
            merged.c.created_at.desc(),
            merged.c.id.desc(),
        )
        .limit(cap)
    )


class AssistantService:
    """
    CRUD, bounded summary reads, and relationship utilities for Assistant.
    """

    RELATIONSHIP_FIELDS = {"users", "vector_stores"}

    @staticmethod
    def _extract_ids(raw_list: list[Any]) -> list[str]:
        ids: list[str] = []
        for item in raw_list or []:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict) and "id" in item:
                ids.append(item["id"])
        return ids

    def __init__(self):
        self.client = Entity()

    # ────────────────────────────────────────────────
    # Ownership guard
    # ────────────────────────────────────────────────

    @staticmethod
    def _assert_owner(db_asst: Assistant, user_id: str) -> None:
        """
        Raise 403 if user_id is not the canonical owner of db_asst.

        Back-fill window: if owner_id is still NULL (migration not yet
        settled) we fall back to checking the many-to-many relationship.
        Once the back-fill follow-up migration sets owner_id NOT NULL,
        the fallback branch can be removed.
        """
        if db_asst.owner_id is not None:
            if db_asst.owner_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to modify this assistant.",
                )
        else:
            associated_ids = {u.id for u in db_asst.users}
            if user_id not in associated_ids:
                raise HTTPException(
                    status_code=403,
                    detail="You do not have permission to modify this assistant.",
                )

    # ────────────────────────────────────────────────
    # Bounded summary reads
    # ────────────────────────────────────────────────

    @staticmethod
    def list_summaries(
        db,
        user_id: str,
        filters: SummaryFilters,
    ) -> SummaryPage:
        """Return a bounded page of active assistants visible to this user.
        Apply ownership, filtering, and cursor pagination in SQL before
        serializing summary fields. The caller owns the database session.
        """
        rows = db.execute(summary_query(user_id, filters)).mappings().all()

        has_more = len(rows) > filters.limit

        items = [AssistantSummary.model_validate(row) for row in rows[: filters.limit]]

        next_cursor = (
            encode_cursor(items[-1].created_at, items[-1].id) if has_more else None
        )

        return SummaryPage(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    # ────────────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────────────

    def create_assistant(
        self,
        assistant: validator.AssistantCreate,
        user_id: str,
    ) -> validator.AssistantRead:
        with SessionLocal() as db:
            assistant_id = (
                assistant.id or UtilsInterface.IdentifierService.generate_assistant_id()
            )

            existing = db.query(Assistant).filter(Assistant.id == assistant_id).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Assistant with ID '{assistant_id}' already exists",
                )

            db_assistant = Assistant(
                id=assistant_id,
                object="assistant",
                created_at=int(time.time()),
                name=assistant.name,
                description=assistant.description,
                model=assistant.model,
                instructions=assistant.instructions,
                tool_configs=assistant.tools,
                tool_resources=assistant.tool_resources,
                meta_data=assistant.meta_data,
                top_p=assistant.top_p,
                temperature=assistant.temperature,
                response_format=assistant.response_format,
                max_tokens=assistant.max_tokens,
                max_turns=assistant.max_turns,
                agent_mode=assistant.agent_mode,
                web_access=assistant.web_access,
                deep_research=assistant.deep_research,
                engineer=assistant.engineer,
                decision_telemetry=assistant.decision_telemetry,
                owner_id=user_id,
                deleted_at=None,
            )

            db.add(db_assistant)
            db.commit()
            db.refresh(db_assistant)
            return self.map_to_read_model(db_assistant)

    def retrieve_assistant(
        self,
        assistant_id: str,
        user_id: str,
    ) -> validator.AssistantRead:
        """
        Fetch an assistant and enforce ownership.

        Use this for all HTTP-boundary calls where the caller must be the
        owner of the assistant. Raises 403 if user_id does not own the
        record, 404 if the assistant does not exist or is soft-deleted.
        """
        with SessionLocal() as db:
            db_asst = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.deleted_at.is_(None),
                )
                .first()
            )
            if not db_asst:
                raise HTTPException(
                    status_code=404,
                    detail="Assistant not found",
                )

            self._assert_owner(db_asst, user_id)

            return self.map_to_read_model(db_asst)

    def retrieve_assistant_internal(
        self,
        assistant_id: str,
    ) -> validator.AssistantRead:
        """
        Fetch an assistant WITHOUT an ownership check.

        FOR INTERNAL / TRUSTED CALLERS ONLY — do not expose this through any
        HTTP router. Intended for services that operate inside the trust
        boundary (e.g. NativeExecutionService, OrchestratorCore) where
        ownership was already verified at the point the run was created.

        Raises 404 if the assistant does not exist or is soft-deleted.
        """
        with SessionLocal() as db:
            db_asst = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.deleted_at.is_(None),
                )
                .first()
            )
            if not db_asst:
                raise HTTPException(
                    status_code=404,
                    detail="Assistant not found",
                )

            return self.map_to_read_model(db_asst)

    def update_assistant(
        self,
        assistant_id: str,
        assistant_update: validator.AssistantUpdate,
        user_id: str,
    ) -> validator.AssistantRead:
        try:
            cache = get_sync_invalidator()
            cache.invalidate_sync(assistant_id)
            logging_utility.info(f"Invalidated cache for assistant {assistant_id}")
        except Exception as e:
            logging_utility.error(f"Failed to invalidate cache: {e}")

        with SessionLocal() as db:
            db_asst = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.deleted_at.is_(None),
                )
                .first()
            )
            if not db_asst:
                raise HTTPException(404, "Assistant not found")

            self._assert_owner(db_asst, user_id)

            data = assistant_update.model_dump(exclude_unset=True)

            for key, val in data.items():
                if key not in self.RELATIONSHIP_FIELDS and key != "tools":
                    setattr(db_asst, key, val)

            if "tools" in data:
                managed_names = {
                    provider_name
                    for (provider_name,) in (
                        db.query(AssistantMcpTool.provider_name)
                        .filter(
                            AssistantMcpTool.assistant_id == assistant_id,
                            AssistantMcpTool.enabled.is_(True),
                        )
                        .all()
                    )
                }

                try:
                    db_asst.tool_configs = merge_user_tools_preserving_managed(
                        list(db_asst.tool_configs or []),
                        data["tools"],
                        managed_names,
                    )
                except ValueError as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=str(exc),
                    ) from exc

            if "users" in data:
                db_asst.users = (
                    db.query(User)
                    .filter(User.id.in_(self._extract_ids(data["users"])))
                    .all()
                )

            if "vector_stores" in data:
                db_asst.vector_stores = (
                    db.query(VectorStore)
                    .filter(
                        VectorStore.id.in_(self._extract_ids(data["vector_stores"]))
                    )
                    .all()
                )

            db.commit()
            db.refresh(db_asst)
            return self.map_to_read_model(db_asst)

    def list_assistants_by_user(
        self,
        user_id: str,
    ) -> List[validator.AssistantRead]:
        """
        Return all active assistants belonging to user_id.

        Ownership is determined by the owner_id column (set at creation).
        We also union in any assistants linked via the legacy many-to-many
        user.assistants relationship so that records created before the
        owner_id migration are still visible.
        """
        with SessionLocal() as db:
            owned_by_column = (
                db.query(Assistant)
                .filter(
                    Assistant.owner_id == user_id,
                    Assistant.deleted_at.is_(None),
                )
                .all()
            )

            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(404, "User not found")

            owned_ids = {a.id for a in owned_by_column}
            legacy_only = [
                a
                for a in user.assistants
                if a.id not in owned_ids and a.owner_id is None and a.deleted_at is None
            ]

            all_assistants = owned_by_column + legacy_only
            return [self.map_to_read_model(a) for a in all_assistants]

    # ────────────────────────────────────────────────
    # DELETE (GDPR Compliant)
    # ────────────────────────────────────────────────

    def delete_assistant(
        self,
        assistant_id: str,
        user_id: str,
        permanent: bool = False,
    ) -> None:
        try:
            cache = get_sync_invalidator()
            cache.invalidate_sync(assistant_id)
        except Exception as e:
            logging_utility.error(f"Failed to invalidate cache during delete: {e}")

        with SessionLocal() as db:
            db_asst = db.query(Assistant).filter(Assistant.id == assistant_id).first()

            if not db_asst:
                raise HTTPException(404, "Assistant not found")

            self._assert_owner(db_asst, user_id)

            if permanent:
                logging_utility.warning(
                    f"PERMANENTLY deleting assistant {assistant_id}"
                )
                db_asst.users = []
                db_asst.vector_stores = []
                db.delete(db_asst)
                db.commit()
            else:
                if db_asst.deleted_at is not None:
                    raise HTTPException(404, "Assistant not found")

                logging_utility.info(f"Soft deleting assistant {assistant_id}")
                db_asst.deleted_at = int(time.time())
                db.commit()

    # ────────────────────────────────────────────────
    # ASSOCIATIONS
    # ────────────────────────────────────────────────

    def associate_assistant_with_user(
        self,
        user_id: str,
        assistant_id: str,
    ) -> None:
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail="User not found",
                )

            db_asst = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.deleted_at.is_(None),
                )
                .first()
            )
            if not db_asst:
                raise HTTPException(
                    status_code=404,
                    detail="Assistant not found",
                )

            if db_asst not in user.assistants:
                user.assistants.append(db_asst)
                db.commit()
                logging_utility.info(
                    f"Associated assistant {assistant_id} with user {user_id}"
                )

    def disassociate_assistant_from_user(
        self,
        user_id: str,
        assistant_id: str,
    ) -> None:
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail="User not found",
                )

            db_asst = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.deleted_at.is_(None),
                )
                .first()
            )
            if not db_asst:
                raise HTTPException(
                    status_code=404,
                    detail="Assistant not found",
                )

            if db_asst in user.assistants:
                user.assistants.remove(db_asst)
                db.commit()
                logging_utility.info(
                    f"Disassociated assistant {assistant_id} from user {user_id}"
                )

    # ────────────────────────────────────────────────
    # Mapper
    # ────────────────────────────────────────────────

    def map_to_read_model(
        self,
        db_asst: Assistant,
    ) -> validator.AssistantRead:
        data = db_asst.__dict__.copy()
        data.pop("_sa_instance_state", None)
        data["tools"] = db_asst.tool_configs or []
        return validator.AssistantRead.model_validate(data)
