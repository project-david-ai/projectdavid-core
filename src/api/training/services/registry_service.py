"""
registry_service.py

Centralized registry for base model catalog management.

Responsibilities:
  - Registering base models with clean prefixed IDs
  - Decoupling the HuggingFace model path from the primary key
  - Providing idempotent upsert semantics for re-registration
  - Listing and retrieving catalog entries

Usage:
    registry = RegistryService(db)
    model = registry.register_base_model(
        hf_model_id="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
        name="Qwen2.5 1.5B Instruct (Unsloth 4bit)",
        family="qwen",
        parameter_count="1.5B",
    )
    print(model.id)       # bm_abc123...
    print(model.endpoint) # unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit
"""

import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.training.models.models import BaseModel

logger = LoggingUtility()


class RegistryService:
    """
    Service layer for the base model catalog.

    All mutating operations go through this class to ensure consistent
    ID generation, idempotency, and audit logging.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_base_model(
        self,
        hf_model_id: str,
        name: str,
        family: Optional[str] = None,
        parameter_count: Optional[str] = None,
        is_multimodal: bool = False,
    ) -> BaseModel:
        """
        Register a base model in the catalog.

        Generates a clean prefixed ID (bm_...) and stores the HuggingFace
        model path in the `endpoint` field. This decouples the primary key
        from the HF path, avoiding slash-in-URL routing issues on activation.

        Idempotent: if a model with the same endpoint is already registered,
        the existing record is returned without modification.

        Args:
            hf_model_id:      HuggingFace model path, e.g.
                              'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit'
            name:             Human-readable display name.
            family:           Model family, e.g. 'qwen', 'llama', 'mistral'.
            parameter_count:  Parameter count string, e.g. '1.5B', '7B'.
            is_multimodal:    True if the model accepts image inputs.

        Returns:
            The existing or newly created BaseModel ORM instance.
        """
        existing = self.db.query(BaseModel).filter(BaseModel.endpoint == hf_model_id).first()
        if existing:
            logger.info(
                "RegistryService: base model already registered — returning existing. "
                "id=%s endpoint=%s",
                existing.id,
                existing.endpoint,
            )
            return existing

        model_id = IdentifierService.generate_prefixed_id("bm")
        base = BaseModel(
            id=model_id,
            name=name,
            family=family,
            parameter_count=parameter_count,
            is_multimodal=is_multimodal,
            endpoint=hf_model_id,
            created_at=int(time.time()),
        )
        self.db.add(base)
        self.db.commit()
        self.db.refresh(base)

        logger.info(
            "RegistryService: registered base model. id=%s endpoint=%s",
            base.id,
            base.endpoint,
        )
        return base

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, model_id: str) -> BaseModel:
        """
        Fetch a base model by its prefixed ID (bm_...).

        Raises 404 if not found.
        """
        base = self.db.query(BaseModel).filter(BaseModel.id == model_id).first()
        if not base:
            raise HTTPException(
                status_code=404,
                detail=f"Base model '{model_id}' not found in registry.",
            )
        return base

    def get_by_endpoint(self, hf_model_id: str) -> BaseModel:
        """
        Fetch a base model by its HuggingFace path / endpoint string.

        Useful for backward compatibility when callers pass the HF path
        instead of the bm_... ID.

        Raises 404 if not found.
        """
        base = self.db.query(BaseModel).filter(BaseModel.endpoint == hf_model_id).first()
        if not base:
            raise HTTPException(
                status_code=404,
                detail=f"No base model registered for endpoint '{hf_model_id}'.",
            )
        return base

    def resolve(self, model_ref: str) -> BaseModel:
        """
        Resolve a model reference to a BaseModel record.

        Accepts either:
          - A prefixed ID:  'bm_abc123...'
          - An HF path:     'unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit'

        Raises 404 if neither lookup succeeds.
        """
        if model_ref.startswith("bm_"):
            return self.get_by_id(model_ref)
        return self.get_by_endpoint(model_ref)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_base_models(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[BaseModel]:
        """
        Return a paginated list of all registered base models.
        """
        return (
            self.db.query(BaseModel)
            .order_by(BaseModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def deregister_base_model(self, model_id: str) -> dict:
        """
        Remove a base model from the catalog by prefixed ID.

        Hard delete — use with caution if active deployments reference
        this model.
        """
        base = self.get_by_id(model_id)
        self.db.delete(base)
        self.db.commit()
        logger.info("RegistryService: deregistered base model. id=%s", model_id)
        return {"status": "deleted", "model_id": model_id}
