# src/api/entities_api/models/models.py
from enum import Enum as PyEnum

from passlib.context import CryptContext
from projectdavid_common import ValidationInterface
from projectdavid_common.projectdavid_orm.base import Base
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_orm.ormInterface import (Action, ApiKey, Assistant, AuditLog,
                                           File, FileStorage, Message, Run,
                                           Sandbox, Thread, User, VectorStore,
                                           VectorStoreFile)
from sqlalchemy import Column, ForeignKey, String, Table

logger = LoggingUtility()

validation = ValidationInterface

# --- Association Tables ---

thread_participants = Table(
    "thread_participants",
    Base.metadata,
    Column("thread_id", String(64), ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", String(64), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)

user_assistants = Table(
    "user_assistants",
    Base.metadata,
    Column("user_id", String(64), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "assistant_id",
        String(64),
        ForeignKey("assistants.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# --- Enums & Context ---


class StatusEnum(PyEnum):
    deleted = "deleted"
    active = "active"
    queued = "queued"
    in_progress = "in_progress"
    pending_action = "action_required"
    completed = "completed"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"
    pending = "pending"
    processing = "processing"
    expired = "expired"
    retrying = "retrying"


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- Core Models ---

ApiKey=ApiKey
User=User
# ───────────────────────────────────────────────
#  AUDIT LOGGING (GDPR & Enterprise Compliance)
# ───────────────────────────────────────────────
AuditLog=AuditLog
Thread=Thread
Message=Message
Run=Run
Assistant=Assistant
Action=Action
Sandbox=Sandbox
File=File
FileStorage=FileStorage
VectorStore=VectorStore
VectorStoreFile=VectorStoreFile




