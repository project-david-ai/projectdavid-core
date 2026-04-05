# src/api/entities_api/models/models.py
"""
Model Aliases — Thin Re-export Layer
=====================================

The canonical SQLAlchemy ORM model definitions for this platform have been
extracted into a dedicated shared package:

    Package : projectdavid-orm
    Repo    : https://github.com/project-david-ai/projectdavid-orm
    Module  : projectdavid_orm.ormInterface

This file exists solely to preserve backwards-compatible imports across the
entities_api codebase. All routers, services, and dependencies that import
from ``src.api.entities_api.models.models`` continue to work without change.

DO NOT define new models here. Any schema changes must be made in
``projectdavid-orm`` and a corresponding Alembic migration must be written
in this repo under ``migrations/versions/``.

Import map
----------
    entities_api.models.models.User          → projectdavid_orm.ormInterface.User
    entities_api.models.models.ApiKey        → projectdavid_orm.ormInterface.ApiKey
    entities_api.models.models.Assistant     → projectdavid_orm.ormInterface.Assistant
    entities_api.models.models.Thread        → projectdavid_orm.ormInterface.Thread
    entities_api.models.models.Message       → projectdavid_orm.ormInterface.Message
    entities_api.models.models.Run           → projectdavid_orm.ormInterface.Run
    entities_api.models.models.Action        → projectdavid_orm.ormInterface.Action
    entities_api.models.models.Sandbox       → projectdavid_orm.ormInterface.Sandbox
    entities_api.models.models.File          → projectdavid_orm.ormInterface.File
    entities_api.models.models.FileStorage   → projectdavid_orm.ormInterface.FileStorage
    entities_api.models.models.VectorStore   → projectdavid_orm.ormInterface.VectorStore
    entities_api.models.models.VectorStoreFile → projectdavid_orm.ormInterface.VectorStoreFile
    entities_api.models.models.AuditLog      → projectdavid_orm.ormInterface.AuditLog
"""
from projectdavid_orm import (
    Action,
    ApiKey,
    Assistant,
    AuditLog,
    Base,
    File,
    FileStorage,
    Message,
    OrmInterface,
    Run,
    Sandbox,
    Thread,
    User,
    VectorStore,
    VectorStoreFile,
)

__all__ = [
    "Action",
    "ApiKey",
    "Assistant",
    "AuditLog",
    "File",
    "FileStorage",
    "Message",
    "Run",
    "Sandbox",
    "Thread",
    "User",
    "VectorStore",
    "VectorStoreFile",
]
