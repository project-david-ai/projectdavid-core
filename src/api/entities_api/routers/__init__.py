# __init__.py
from fastapi import APIRouter

from src.api.entities_api.routers.actions_router import \
    router as actions_router
from src.api.entities_api.routers.admin_router import \
    admin_router as admin_router
from src.api.entities_api.routers.api_key_router import \
    router as api_key_router
from src.api.entities_api.routers.assistants_router import \
    router as assistants_router
from src.api.entities_api.routers.files_router import router as files_router
from src.api.entities_api.routers.inference_router import \
    router as inference_router
from src.api.entities_api.routers.messages_router import \
    router as messages_router
from src.api.entities_api.routers.runs_router import router as runs_router
from src.api.entities_api.routers.sandbox_auth_router import \
    router as sandbox_auth_router
from src.api.entities_api.routers.threads_router import \
    router as threads_router
from src.api.entities_api.routers.tools_router import router as tools_router
from src.api.entities_api.routers.users_router import router as users_router
from src.api.entities_api.routers.vectors_router import \
    router as vectors_router

api_router = APIRouter()
api_router.include_router(inference_router, tags=["Inference"])
api_router.include_router(threads_router, tags=["Threads"])
api_router.include_router(users_router, tags=["Users"])
api_router.include_router(runs_router, tags=["Runs"])
api_router.include_router(assistants_router, tags=["Assistants"])
api_router.include_router(messages_router, tags=["Messages"])
api_router.include_router(actions_router, tags=["Actions"])
api_router.include_router(files_router, tags=["Files"])
api_router.include_router(vectors_router, tags=["Vector Stores"])
api_router.include_router(api_key_router, tags=["API Keys"])
api_router.include_router(admin_router, tags=["Admin"])
api_router.include_router(tools_router, tags=["Tools: Web Browsing"])
api_router.include_router(sandbox_auth_router, tags=["Tools: Sandbox Authorization"])
