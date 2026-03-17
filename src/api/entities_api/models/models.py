# src/api/entities_api/models/models.py
import secrets
import time
from datetime import datetime
from enum import Enum as PyEnum

from passlib.context import CryptContext
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import declarative_base, joinedload, relationship

logger = LoggingUtility()
Base = declarative_base()
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


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    key_name = Column(String(100), nullable=True)
    hashed_key = Column(String(255), unique=True, nullable=False, index=True)
    prefix = Column(String(8), unique=True, nullable=False)
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    user = relationship("User", back_populates="api_keys")
    __table_args__ = (Index("idx_apikey_user_id_active", "user_id", "is_active"),)

    @staticmethod
    def generate_key(prefix="sk_"):
        return f"{prefix}{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_key(key: str) -> str:
        return pwd_context.hash(key)

    def verify_key(self, plain_key: str) -> bool:
        return pwd_context.verify(plain_key, self.hashed_key)


class User(Base):
    __tablename__ = "users"
    id = Column(
        String(64),
        primary_key=True,
        index=True,
        comment="Internal unique identifier for the user (e.g., user_...)",
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        server_default="0",
        comment="Flag indicating administrative privileges",
    )
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=True,
        comment="Primary email address, potentially verified by OAuth provider",
    )
    email_verified = Column(
        Boolean,
        default=False,
        nullable=True,
        comment="Whether the email address has been verified",
    )

    full_name = Column(String(255), nullable=True, comment="User's full display name")
    given_name = Column(String(128), nullable=True, comment="First name")
    family_name = Column(String(128), nullable=True, comment="Last name")
    picture_url = Column(Text, nullable=True, comment="URL to the user's profile picture")

    oauth_provider = Column(
        String(50),
        nullable=True,
        index=True,
        comment="Name of the OAuth provider (e.g., 'google', 'github', 'local')",
    )
    provider_user_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="The unique ID assigned by the OAuth provider",
    )

    # Relationships
    api_keys = relationship(
        "ApiKey", back_populates="user", cascade="all, delete-orphan", lazy="select"
    )

    threads = relationship(
        "Thread",
        secondary=thread_participants,
        back_populates="participants",
        lazy="select",
    )

    assistants = relationship(
        "Assistant", secondary=user_assistants, back_populates="users", lazy="select"
    )
    sandboxes = relationship(
        "Sandbox", back_populates="user", cascade="all, delete-orphan", lazy="select"
    )
    vector_stores = relationship("VectorStore", back_populates="user", lazy="select")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan", lazy="select")
    runs = relationship("Run", back_populates="user", lazy="select")

    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")

    __table_args__ = (
        UniqueConstraint("oauth_provider", "provider_user_id", name="uq_user_oauth_provider_id"),
        Index("idx_user_email", "email"),
        Index("idx_user_is_admin", "is_admin"),
    )


# ───────────────────────────────────────────────
#  AUDIT LOGGING (GDPR & Enterprise Compliance)
# ───────────────────────────────────────────────
class AuditLog(Base):
    """
    Immutable record of system actions.
    Supports GDPR compliance by tracking deletions and administrative access.

    FK: user_id → SET NULL on user deletion.
    Audit records must survive user erasure — they are the compliance trail.
    """

    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)

    # Who performed the action.
    # Nullable: SET NULL when the user is erased (record survives, identity is anonymised).
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # What was done
    action = Column(
        String(32),
        nullable=False,
        index=True,
        comment="e.g. CREATE, UPDATE, DELETE, HARD_DELETE, ERASE",
    )

    # What entity was affected
    entity_type = Column(
        String(64), nullable=False, index=True, comment="e.g. Assistant, User, Thread"
    )
    entity_id = Column(String(64), nullable=False, index=True)

    # Context
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)

    # Detailed Payload (snapshot of changes)
    details = Column(
        JSON, nullable=True, comment="Stores before/after state or reasoning for action"
    )

    # Relationship
    user = relationship("User", back_populates="audit_logs")


class Thread(Base):
    __tablename__ = "threads"
    id = Column(String(64), primary_key=True, index=True)
    created_at = Column(Integer, nullable=False)
    meta_data = Column(JSON, nullable=False, default={})
    object = Column(String(64), nullable=False)
    tool_resources = Column(JSON, nullable=False, default={})
    participants = relationship("User", secondary=thread_participants, back_populates="threads")

    # SET NULL on user deletion — thread record is preserved, ownership is cleared.
    # Application-level erase_user() handles physical cleanup of thread content.
    owner_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Canonical creator/owner of this thread. Used for row-level access control.",
    )
    owner = relationship("User", foreign_keys=[owner_id], lazy="select")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String(64), primary_key=True, index=True)
    assistant_id = Column(String(64), index=True)
    attachments = Column(JSON, default=[])
    completed_at = Column(Integer, nullable=True)

    # --- The "Brain" Output ---
    content = Column(Text(length=4294967295), nullable=False)
    reasoning = Column(
        Text(length=4294967295),
        nullable=True,
        comment="Stores the internal 'thinking' or reasoning tokens from the model.",
    )

    # --- Native Tool Metadata (Critical for Stage 2/Level 3) ---
    tool_calls = Column(
        JSON,
        nullable=True,
        comment="Stores the native JSON array of tool calls generated by the assistant.",
    )
    tool_call_id = Column(
        String(64),
        nullable=True,
        comment="For messages with role='tool', this links back to the specific tool_call_id.",
    )

    created_at = Column(Integer, nullable=False)
    incomplete_at = Column(Integer, nullable=True)
    incomplete_details = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=False, default={})
    object = Column(String(64), nullable=False)
    role = Column(String(32), nullable=False)
    run_id = Column(String(64), nullable=True)

    # Kept as simple String, no longer linked to deleted Tool table
    tool_id = Column(String(64), nullable=True)

    status = Column(String(32), nullable=True)

    # NOTE: thread_id is intentionally stored as a plain string with no FK constraint.
    # Orphaned messages (after thread deletion) are handled by:
    #   a) application-level erase_user() during right-to-erasure
    #   b) purge_orphaned_threads daemon (periodic background cleanup)
    thread_id = Column(String(64), nullable=False)
    sender_id = Column(String(64), nullable=True)


class Run(Base):
    __tablename__ = "runs"

    # CASCADE: runs are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    user = relationship("User", back_populates="runs")

    id = Column(String(64), primary_key=True)
    assistant_id = Column(String(64), nullable=False)

    # timestamps as epoch seconds (ints)
    cancelled_at = Column(Integer, nullable=True)
    completed_at = Column(Integer, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    expires_at = Column(Integer, nullable=True)
    failed_at = Column(Integer, nullable=True)
    started_at = Column(Integer, nullable=True)

    incomplete_details = Column(String(256), nullable=True)

    instructions = Column(Text, nullable=True)
    last_error = Column(String(256), nullable=True)
    max_completion_tokens = Column(Integer, nullable=True)
    max_prompt_tokens = Column(Integer, nullable=True)

    meta_data = Column(MutableDict.as_mutable(JSON), nullable=True, default=dict)

    model = Column(String(64), nullable=True)
    object = Column(String(64), nullable=False)
    parallel_tool_calls = Column(Boolean, default=False)
    required_action = Column(String(256), nullable=True)
    response_format = Column(String(64), nullable=True)

    status = Column(SAEnum(validation.StatusEnum), nullable=False)

    # NOTE: thread_id stored as plain string — no FK constraint (mirrors Message.thread_id).
    thread_id = Column(String(64), nullable=False)
    tool_choice = Column(String(64), nullable=True)

    tools = Column(JSON, nullable=True)

    # --- Agentic Behavior State (Level 3) ---
    current_turn = Column(
        Integer,
        default=0,
        server_default="0",
        comment="The current iteration count of the autonomous loop.",
    )
    max_turns = Column(
        Integer,
        default=1,
        server_default="1",
        comment="Snapshot of Assistant.max_turns at run creation.",
    )
    agent_mode = Column(
        String(32),
        default="standard",
        server_default="standard",
        comment="Determines execution logic: 'standard' (Level 2) or 'autonomous' (Level 3).",
    )

    truncation_strategy = Column(
        String(16),
        nullable=True,
        default="auto",
        server_default="auto",
    )

    usage = Column(JSON, nullable=True)
    temperature = Column(Integer, nullable=True)
    top_p = Column(Integer, nullable=True)
    tool_resources = Column(JSON, nullable=True)

    actions = relationship("Action", back_populates="run")


class Assistant(Base):
    __tablename__ = "assistants"

    id = Column(String(64), primary_key=True, index=True)
    object = Column(String(64), nullable=False)
    created_at = Column(Integer, nullable=False)

    name = Column(String(128), nullable=False)
    description = Column(String(256))
    model = Column(String(64))
    instructions = Column(Text)

    tool_configs = Column(JSON)
    meta_data = Column(JSON)
    top_p = Column(Integer)
    temperature = Column(Integer)
    response_format = Column(String(64))

    tool_resources = Column(
        JSON,
        nullable=True,
        comment='Resource map keyed by tool type, e.g. {"file_search": {"vector_store_ids": ["vs_123","vs_456"]}}',
    )

    # SET NULL on user deletion — assistant record is preserved (may be shared with
    # other users via user_assistants). Application-level erase_user() decides
    # whether to soft-delete exclusively-owned assistants.
    owner_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment=(
            "Canonical owner of this assistant. "
            "Primary key for row-level access filtering. "
            "Separate from the many-to-many (user_assistants) which handles sharing."
        ),
    )

    # --- Agentic Behavior Extensions (Level 3) ---
    max_turns = Column(
        Integer,
        default=1,
        server_default="1",
        comment="Max number of iterative loops for Level 3 agency. 1 = Standard Level 2 (ReAct).",
    )
    agent_mode = Column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="False = Standard (Level 2), True = Autonomous (Level 3).",
    )
    web_access = Column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="Enable live web search and browsing capabilities.",
    )
    deep_research = Column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="Enable deep research capabilities.",
    )
    engineer = Column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="Enable network engineering capabilities and inventory map access.",
    )
    decision_telemetry = Column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="If True, captures detailed reasoning payloads and confidence scores.",
    )

    # --- GDPR / Lifecycle Management ---
    deleted_at = Column(
        Integer,
        nullable=True,
        default=None,
        comment="Unix timestamp of soft-deletion. If present, entity is in 'Recycle Bin'.",
    )

    # --- Relationships ---
    owner = relationship(
        "User",
        foreign_keys=[owner_id],
        lazy="select",
    )

    users = relationship(
        "User",
        secondary="user_assistants",
        back_populates="assistants",
        lazy="select",
    )


class Action(Base):
    __tablename__ = "actions"

    id = Column(String(64), primary_key=True, index=True)

    # CASCADE: actions are deleted when their parent run is deleted.
    run_id = Column(
        String(64),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=True,
    )

    # --- Agentic Tracking (Level 3) ---
    tool_call_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="The unique ID linking this action to a specific LLM tool request.",
    )

    tool_name = Column(
        String(64),
        nullable=True,
        comment="The name of the function/tool to be executed.",
    )

    turn_index = Column(
        Integer,
        default=0,
        nullable=True,
        comment="The iteration of the autonomous loop.",
    )

    decision_payload = Column(
        JSON,
        nullable=True,
        comment="The full structured reasoning object (reason, policy, etc) preceding the tool call.",
    )

    confidence_score = Column(
        Float,
        nullable=True,
        index=True,
        comment="Extracted confidence score (0.0-1.0) to allow fast querying of 'uncertain' agent actions.",
    )

    triggered_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    status = Column(String(64), nullable=True)
    function_args = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)

    # --- Relationships ---
    run = relationship("Run", back_populates="actions")

    @staticmethod
    def get_full_action_query(session):
        return session.query(Action).options(joinedload(Action.run))


class Sandbox(Base):
    __tablename__ = "sandboxes"
    id = Column(String(64), primary_key=True, index=True)

    # CASCADE: sandboxes are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), nullable=False, default="active")
    config = Column(JSON, nullable=True)
    user = relationship("User", back_populates="sandboxes")


class File(Base):
    __tablename__ = "files"
    id = Column(String(64), primary_key=True, index=True)
    object = Column(String(64), nullable=False, default="file")
    bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    filename = Column(String(256), nullable=False)
    purpose = Column(String(64), nullable=False)
    mime_type = Column(String(255))

    # ── GDPR / Lifecycle ────────────────────────────────────────────
    deleted_at = Column(
        Integer,
        nullable=True,
        default=None,
        index=True,
        comment="Unix timestamp of soft-deletion. Non-null = file is in Recycle Bin.",
    )

    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    user = relationship("User", back_populates="files")
    storage_locations = relationship(
        "FileStorage", back_populates="file", cascade="all, delete-orphan"
    )


class FileStorage(Base):
    __tablename__ = "file_storage"
    id = Column(Integer, primary_key=True, autoincrement=True)

    # CASCADE: storage records are deleted when their parent file is deleted.
    file_id = Column(
        String(64),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
    )
    storage_system = Column(
        String(64),
        nullable=False,
        default="samba",
        comment="Storage system type (samba, s3, etc.)",
    )
    storage_path = Column(
        String(512),
        nullable=False,
        comment="Path to file in storage system (relative to share root)",
    )
    is_primary = Column(
        Boolean,
        default=True,
        comment="Indicates if this is the primary storage location",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="When this storage entry was created",
    )
    file = relationship("File", back_populates="storage_locations")
    __table_args__ = (Index("idx_file_storage_file_id", "file_id"),)


class BatfishSnapshot(Base):
    __tablename__ = "batfish_snapshots"

    id = Column(
        String(64),
        primary_key=True,
        index=True,
        comment="Opaque snapshot ID returned to caller e.g. snap_abc123",
    )

    snapshot_name = Column(
        String(128),
        nullable=False,
        comment="Caller-supplied label e.g. 'incident_001'",
    )

    snapshot_key = Column(
        String(256),
        nullable=False,
        unique=True,
        index=True,
        comment="Namespaced isolation key: {user_id}_{id}",
    )

    # CASCADE: snapshots are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    configs_root = Column(String(512), nullable=True)

    device_count = Column(Integer, default=0, nullable=False)
    devices = Column(
        JSON,
        default=list,
        nullable=False,
        comment="List of hostnames ingested into this snapshot",
    )

    status = Column(
        SAEnum(StatusEnum),
        nullable=False,
        default=StatusEnum.pending,
    )

    error_message = Column(Text, nullable=True)

    created_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    updated_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    last_ingested_at = Column(BigInteger, nullable=True)

    user = relationship("User", lazy="select")

    __table_args__ = (
        UniqueConstraint("user_id", "snapshot_name", name="uq_batfish_user_snapshot_name"),
        Index("idx_batfish_user_id", "user_id"),
        Index("idx_batfish_status", "status"),
    )


class VectorStore(Base):
    __tablename__ = "vector_stores"
    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False, unique=False)

    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    collection_name = Column(String(128), nullable=False, unique=True)
    vector_size = Column(Integer, nullable=False)
    distance_metric = Column(String(32), nullable=False)
    created_at = Column(BigInteger, default=lambda: int(datetime.now().timestamp()))
    updated_at = Column(BigInteger, onupdate=lambda: int(datetime.now().timestamp()))
    status = Column(SAEnum(StatusEnum), nullable=False)
    config = Column(JSON, nullable=True)
    file_count = Column(Integer, default=0, nullable=False)

    # ── GDPR / Lifecycle ────────────────────────────────────────────────────
    deleted_at = Column(
        Integer,
        nullable=True,
        default=None,
        index=True,
        comment="Unix timestamp of soft-deletion. Non-null = store is in Recycle Bin.",
    )

    user = relationship("User", back_populates="vector_stores", lazy="select")
    files = relationship(
        "VectorStoreFile",
        back_populates="vector_store",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )


class VectorStoreFile(Base):
    __tablename__ = "vector_store_files"
    id = Column(String(64), primary_key=True, index=True)

    # CASCADE: vector store files are deleted when their parent store is deleted.
    vector_store_id = Column(
        String(64),
        ForeignKey("vector_stores.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_name = Column(String(256), nullable=False)
    file_path = Column(String(1024), nullable=False)
    processed_at = Column(Integer, nullable=True)
    status = Column(SAEnum(StatusEnum), default=StatusEnum.queued)
    error_message = Column(Text, nullable=True)
    meta_data = Column(JSON, nullable=True)
    vector_store = relationship("VectorStore", back_populates="files")


# ─────────────────────────────────────────────────────────────────────────────
#  FINE-TUNING PIPELINE
#
#  Three new tables that power the self-hosted fine-tuning loop:
#
#    Dataset        — registered training datasets (raw files on Samba)
#    TrainingJob    — a fine-tuning run (config, status, lifecycle timestamps)
#    FineTunedModel — a trained model artifact (HF repo or local Samba path)
#
#  Ownership follows the same pattern as VectorStore / File:
#    - user_id FK → CASCADE on user deletion
#    - soft-deletion via deleted_at timestamp
#    - status tracked via existing StatusEnum
# ─────────────────────────────────────────────────────────────────────────────
class Dataset(Base):
    """
    A registered training dataset.

    Raw files live on Samba at `storage_path`.  The Dataset record is the
    registry entry — it tracks format, split sizes, and which training jobs
    have consumed this data.
    """

    __tablename__ = "datasets"

    id = Column(
        String(64),
        primary_key=True,
        index=True,
        comment="Opaque dataset ID e.g. ds_abc123",
    )

    # CASCADE: datasets are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)

    format = Column(
        String(32),
        nullable=False,
        comment="Training format: chatml | alpaca | sharegpt | jsonl",
    )

    storage_path = Column(
        String(512),
        nullable=False,
        comment="Path to the dataset file(s) on Samba, relative to the share root.",
    )

    # Split metadata — populated after dataset preparation
    train_samples = Column(Integer, nullable=True)
    eval_samples = Column(Integer, nullable=True)

    # Free-form config — tokenizer, sequence length, special tokens etc.
    config = Column(JSON, nullable=True)

    status = Column(
        SAEnum(StatusEnum),
        nullable=False,
        default=StatusEnum.pending,
        comment="pending → processing → active → failed",
    )

    created_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    updated_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)

    deleted_at = Column(
        Integer,
        nullable=True,
        default=None,
        index=True,
        comment="Unix timestamp of soft-deletion.",
    )

    # Relationships
    user = relationship("User", lazy="select")
    training_jobs = relationship("TrainingJob", back_populates="dataset", lazy="dynamic")

    __table_args__ = (
        Index("idx_dataset_user_id", "user_id"),
        Index("idx_dataset_status", "status"),
    )


class TrainingJob(Base):
    """
    A fine-tuning job.

    Tracks the full lifecycle of a training run — from queued through to a
    finished model artifact.  The config column stores the full training
    configuration (LoRA rank, learning rate, epochs, etc.) as JSON so it
    can be replayed or diffed.
    """

    __tablename__ = "training_jobs"

    id = Column(
        String(64),
        primary_key=True,
        index=True,
        comment="Opaque job ID e.g. tj_abc123",
    )

    # CASCADE: jobs are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    dataset_id = Column(
        String(64),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source dataset. SET NULL if dataset is deleted — job record is preserved.",
    )

    base_model = Column(
        String(256),
        nullable=False,
        comment="Base model identifier e.g. Qwen/Qwen2.5-7B-Instruct",
    )

    framework = Column(
        String(32),
        nullable=False,
        default="axolotl",
        comment="Training framework: axolotl | unsloth",
    )

    # Full training config — LoRA rank, learning rate, epochs, batch size etc.
    config = Column(
        JSON,
        nullable=True,
        comment="Complete training configuration passed to the training container.",
    )

    status = Column(
        SAEnum(StatusEnum),
        nullable=False,
        default=StatusEnum.queued,
        comment="queued → in_progress → completed | failed | cancelled",
    )

    # Lifecycle timestamps (epoch seconds)
    created_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    started_at = Column(BigInteger, nullable=True)
    completed_at = Column(BigInteger, nullable=True)
    failed_at = Column(BigInteger, nullable=True)

    last_error = Column(Text, nullable=True)

    # Training metrics — populated on completion
    metrics = Column(
        JSON,
        nullable=True,
        comment="Final training metrics: loss, eval_loss, perplexity etc.",
    )

    # Output — path to the raw checkpoint before registry push
    output_path = Column(
        String(512),
        nullable=True,
        comment="Samba path to the training output checkpoint.",
    )

    # Relationships
    user = relationship("User", lazy="select")
    dataset = relationship("Dataset", back_populates="training_jobs", lazy="select")
    fine_tuned_model = relationship(
        "FineTunedModel",
        back_populates="training_job",
        uselist=False,
        lazy="select",
    )

    __table_args__ = (
        Index("idx_trainingjob_user_id", "user_id"),
        Index("idx_trainingjob_status", "status"),
        Index("idx_trainingjob_dataset_id", "dataset_id"),
    )


class FineTunedModel(Base):
    """
    A registered fine-tuned model artifact.

    Created when a TrainingJob completes successfully.  Tracks where the
    model lives (HuggingFace Hub or local Samba path) and whether it is
    currently loaded in vLLM.
    """

    __tablename__ = "fine_tuned_models"

    id = Column(
        String(64),
        primary_key=True,
        index=True,
        comment="Opaque model ID e.g. ftm_abc123",
    )

    # CASCADE: model records are deleted when their owning user is erased.
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    training_job_id = Column(
        String(64),
        ForeignKey("training_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source training job. SET NULL if job is deleted — model record is preserved.",
    )

    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)

    base_model = Column(
        String(256),
        nullable=False,
        comment="Base model this was fine-tuned from.",
    )

    # Where the model lives
    hf_repo = Column(
        String(256),
        nullable=True,
        comment="HuggingFace repository ID e.g. your-org/your-model",
    )

    storage_path = Column(
        String(512),
        nullable=True,
        comment="Local Samba path to model weights (used when not pushed to HF).",
    )

    # Deployment state
    is_active = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="True when this model is currently loaded in vLLM.",
    )

    vllm_model_id = Column(
        String(256),
        nullable=True,
        comment="The VLLM_MODEL value used to serve this model.",
    )

    status = Column(
        SAEnum(StatusEnum),
        nullable=False,
        default=StatusEnum.processing,
        comment="processing → active → failed",
    )

    created_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    updated_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)

    deleted_at = Column(
        Integer,
        nullable=True,
        default=None,
        index=True,
        comment="Unix timestamp of soft-deletion.",
    )

    # Relationships
    user = relationship("User", lazy="select")
    training_job = relationship(
        "TrainingJob",
        back_populates="fine_tuned_model",
        lazy="select",
    )

    __table_args__ = (
        Index("idx_finetunedmodel_user_id", "user_id"),
        Index("idx_finetunedmodel_status", "status"),
        Index("idx_finetunedmodel_is_active", "is_active"),
    )
