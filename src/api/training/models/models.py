"""
Model Aliases — Thin Re-export Layer (Training Domain)
=====================================================

The canonical SQLAlchemy ORM model definitions for training workflows live in:

    Package : projectdavid-orm
    Repo    : https://github.com/project-david-ai/projectdavid-orm
    Module  : projectdavid_orm.ormInterface

This module exists purely for backwards-compatible import stability inside
the training API surface.

Routers, services, and dependency layers importing from:

    src.api.training.models.models

will continue to function without requiring refactors.

⚠️ DO NOT define ORM models here.
All schema evolution must occur inside ``projectdavid-orm`` followed by
a corresponding Alembic migration inside this repository.

Import map
----------
    training.models.models.Dataset         → projectdavid_orm.ormInterface.Dataset
    training.models.models.TrainingJob     → projectdavid_orm.ormInterface.TrainingJob
    training.models.models.FineTunedModel  → projectdavid_orm.ormInterface.FineTunedModel
"""

from projectdavid_orm import (
    BaseModel,
    ComputeNode,
    Dataset,
    FineTunedModel,
    GPUAllocation,
    InferenceDeployment,
    TrainingJob,
)

__all__ = [
    "BaseModel",
    "ComputeNode",
    "Dataset",
    "FineTunedModel",
    "GPUAllocation",
    "InferenceDeployment",
    "TrainingJob",
]
