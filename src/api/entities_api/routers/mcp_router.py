"""Authenticated MCP registration management API."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.mcp_schemas import McpServerAuth
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

router = APIRouter(tags=["MCP"])
validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()


@router.post(
    "/mcp/servers",
    response_model=validator.McpServerRegistrationRead,
)
def create_mcp_server(
    registration: validator.McpServerRegistrationCreate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] registering MCP server")

    return McpRegistrationService().register_server(
        registration,
        user_id=auth_key.user_id,
    )


@router.get(
    "/mcp/servers",
    response_model=list[validator.McpServerRegistrationRead],
)
def list_mcp_servers(
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] listing MCP servers")

    return McpRegistrationService().list_servers(
        user_id=auth_key.user_id,
    )


@router.get(
    "/mcp/servers/{server_id}",
    response_model=validator.McpServerRegistrationRead,
)
def get_mcp_server(
    server_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] reading MCP server " f"{server_id}")

    return McpRegistrationService().get_server(
        server_id=server_id,
        user_id=auth_key.user_id,
    )


@router.patch(
    "/mcp/servers/{server_id}",
    response_model=validator.McpServerRegistrationRead,
)
def update_mcp_server(
    server_id: str,
    registration: validator.McpServerRegistrationUpdate,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] updating MCP server " f"{server_id}")

    return McpRegistrationService().update_server(
        server_id=server_id,
        registration=registration,
        user_id=auth_key.user_id,
        is_protected_key=bool(auth_key.is_protected),
    )


async def _rotation_auth(request: Request) -> McpServerAuth:
    # FastAPI's default validation errors may reflect raw input. Parse this
    # credential-bearing body here and return only a fixed, sanitized error.
    try:
        auth = McpServerAuth.model_validate(await request.json())
    except ValueError:
        raise HTTPException(422, "Invalid MCP bearer token request") from None
    if auth.type != "bearer" or auth.token is None:
        raise HTTPException(422, "Bearer authentication is required")
    return auth


@router.put(
    "/mcp/servers/{server_id}/credential",
    response_model=validator.McpServerRegistrationRead,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["type", "token"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["bearer"]},
                            "token": {
                                "type": "string",
                                "format": "password",
                                "writeOnly": True,
                            },
                        },
                    },
                },
            },
        },
    },
)
def rotate_mcp_server_credential(
    server_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
    auth: McpServerAuth = Depends(_rotation_auth),
):
    if auth.token is None:
        raise HTTPException(422, "Bearer authentication is required")

    return McpRegistrationService().rotate_server_credential(
        server_id=server_id,
        user_id=auth_key.user_id,
        token=auth.token,
        is_protected_key=bool(auth_key.is_protected),
    )


@router.delete(
    "/mcp/servers/{server_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_mcp_server(
    server_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] deleting MCP server " f"{server_id}")

    McpRegistrationService().delete_server(
        server_id=server_id,
        user_id=auth_key.user_id,
        is_protected_key=bool(auth_key.is_protected),
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/mcp/servers/{server_id}/tools",
    response_model=validator.McpToolDiscoveryPageRead,
)
async def discover_mcp_server_tools(
    server_id: str,
    cursor: str | None = None,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] discovering MCP server " f"{server_id}")

    page = await McpRegistrationService().discover_tools(
        user_id=auth_key.user_id,
        server_id=server_id,
        cursor=cursor,
    )

    return validator.McpToolDiscoveryPageRead(
        tools=[
            validator.McpDiscoveredToolRead(
                server_id=tool.server_id,
                remote_name=tool.remote_name,
                canonical_id=tool.canonical_id,
                provider_name=tool.provider_name,
                definition=tool.definition.to_function_tool(),
            )
            for tool in page.tools
        ],
        next_cursor=page.next_cursor,
    )


@router.get(
    "/assistants/{assistant_id}/mcp-tools",
    response_model=list[validator.AssistantMcpToolRead],
)
def list_assistant_mcp_tools(
    assistant_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] listing assistant MCP " f"tools for {assistant_id}"
    )

    return McpRegistrationService().list_assistant_tools(
        assistant_id=assistant_id,
        user_id=auth_key.user_id,
        is_protected_key=bool(auth_key.is_protected),
    )


@router.post(
    "/assistants/{assistant_id}/mcp-tools",
    response_model=list[validator.AssistantMcpToolRead],
)
async def attach_assistant_mcp_tools(
    assistant_id: str,
    attachment: validator.AssistantMcpToolsAttach,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] attaching assistant MCP " f"tools to {assistant_id}"
    )

    return await McpRegistrationService().attach_tools(
        assistant_id=assistant_id,
        attachment=attachment,
        user_id=auth_key.user_id,
        is_protected_key=bool(auth_key.is_protected),
    )


@router.delete(
    "/assistants/{assistant_id}/mcp-tools",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def detach_assistant_mcp_tools(
    assistant_id: str,
    attachment: validator.AssistantMcpToolsDetach,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] detaching assistant MCP " f"tools from {assistant_id}"
    )

    McpRegistrationService().detach_tools(
        assistant_id=assistant_id,
        attachment=attachment,
        user_id=auth_key.user_id,
        is_protected_key=bool(auth_key.is_protected),
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
