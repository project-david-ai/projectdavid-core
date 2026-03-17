import time

from projectdavid_common.projectdavid_orm.base import Base
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy import JSON, BigInteger, Boolean, Column
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(
        String(64), primary_key=True, index=True, comment="Opaque dataset ID e.g. ds_abc123"
    )
    user_id = Column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(
        String(32), nullable=False, comment="Training format: chatml | alpaca | sharegpt | jsonl"
    )
    # Reference to the uploaded file in the core API files table.
    # The actual Samba path is resolved via GET /v1/files/{file_id} at training time.
    file_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Reference to the file_id in the core API files table.",
    )

    # Populated by the training worker after it stages the file for training.
    storage_path = Column(
        String(512),
        nullable=True,
        comment="Resolved Samba path — populated by worker at training time.",
    )

    train_samples = Column(Integer, nullable=True)
    eval_samples = Column(Integer, nullable=True)
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
        Integer, nullable=True, default=None, index=True, comment="Unix timestamp of soft-deletion."
    )

    training_jobs = relationship("TrainingJob", back_populates="dataset", lazy="dynamic")

    __table_args__ = (
        Index("idx_dataset_user_id", "user_id"),
        Index("idx_dataset_status", "status"),
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String(64), primary_key=True, index=True, comment="Opaque job ID e.g. tj_abc123")
    user_id = Column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id = Column(
        String(64),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source dataset. SET NULL if dataset is deleted — job record is preserved.",
    )
    base_model = Column(
        String(256), nullable=False, comment="Base model identifier e.g. Qwen/Qwen2.5-7B-Instruct"
    )
    framework = Column(
        String(32),
        nullable=False,
        default="axolotl",
        comment="Training framework: axolotl | unsloth",
    )
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
    created_at = Column(BigInteger, default=lambda: int(time.time()), nullable=False)
    started_at = Column(BigInteger, nullable=True)
    completed_at = Column(BigInteger, nullable=True)
    failed_at = Column(BigInteger, nullable=True)
    last_error = Column(Text, nullable=True)
    metrics = Column(
        JSON, nullable=True, comment="Final training metrics: loss, eval_loss, perplexity etc."
    )
    output_path = Column(
        String(512), nullable=True, comment="Samba path to the training output checkpoint."
    )

    dataset = relationship("Dataset", back_populates="training_jobs", lazy="select")
    fine_tuned_model = relationship(
        "FineTunedModel", back_populates="training_job", uselist=False, lazy="select"
    )

    __table_args__ = (
        Index("idx_trainingjob_user_id", "user_id"),
        Index("idx_trainingjob_status", "status"),
        Index("idx_trainingjob_dataset_id", "dataset_id"),
    )


class FineTunedModel(Base):
    __tablename__ = "fine_tuned_models"

    id = Column(String(64), primary_key=True, index=True, comment="Opaque model ID e.g. ftm_abc123")
    user_id = Column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
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
    base_model = Column(String(256), nullable=False, comment="Base model this was fine-tuned from.")
    hf_repo = Column(
        String(256), nullable=True, comment="HuggingFace repository ID e.g. your-org/your-model"
    )
    storage_path = Column(String(512), nullable=True, comment="Local Samba path to model weights.")
    is_active = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="True when this model is currently loaded in vLLM.",
    )
    vllm_model_id = Column(
        String(256), nullable=True, comment="The VLLM_MODEL value used to serve this model."
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
        Integer, nullable=True, default=None, index=True, comment="Unix timestamp of soft-deletion."
    )

    training_job = relationship("TrainingJob", back_populates="fine_tuned_model", lazy="select")

    __table_args__ = (
        Index("idx_finetunedmodel_user_id", "user_id"),
        Index("idx_finetunedmodel_status", "status"),
        Index("idx_finetunedmodel_is_active", "is_active"),
    )
