# src/api/entities_api/routers/messages.py
import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from redis.asyncio import Redis

from src.api.entities_api.dependencies import get_api_key, get_redis
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.message_service import MessageService

router = APIRouter()
validator = ValidationInterface()
logging_utility = LoggingUtility()


async def _push_to_redis(redis: Redis, message_obj):
    """Asynchronously pushes message data to Redis list and trims."""
    try:
        key = f"thread:{message_obj.thread_id}:history"
        data = message_obj.model_dump(exclude_unset=True)
        data["role"] = message_obj.role
        data["content"] = message_obj.content if message_obj.content is not None else ""
        payload = json.dumps(data)
        await redis.rpush(key, payload)
        await redis.ltrim(key, -200, -1)
        logging_utility.debug(f"Async Pushed message {message_obj.id} to Redis '{key}'")
    except Exception as e:
        logging_utility.error(
            f"Redis async push failed for {message_obj.id}: {e}", exc_info=True
        )


@router.post("/messages", response_model=ValidationInterface.MessageRead)
async def create_message(
    message: ValidationInterface.MessageCreate,
    redis: Redis = Depends(get_redis),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Creating message in thread {message.thread_id}"
    )
    svc = MessageService()
    try:
        new_message = svc.create_message(message, user_id=auth_key.user_id)
        logging_utility.info(f"Created message ID {new_message.id}")
        await _push_to_redis(redis, new_message)
        return new_message
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Error creating message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.post("/messages/tools", response_model=ValidationInterface.MessageRead)
async def submit_tool_response(
    message: ValidationInterface.MessageCreate,
    redis: Redis = Depends(get_redis),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Submitting tool output for thread {message.thread_id}"
    )
    svc = MessageService()
    try:
        new_message = svc.submit_tool_output(message, user_id=auth_key.user_id)
        logging_utility.info(f"Created tool message ID {new_message.id}")
        await _push_to_redis(redis, new_message)
        return new_message
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Error submitting tool message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get("/messages/{message_id}", response_model=ValidationInterface.MessageRead)
def get_message(
    message_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Retrieving message {message_id}")
    svc = MessageService()
    try:
        return svc.retrieve_message(message_id, user_id=auth_key.user_id)
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Error retrieving message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get(
    "/threads/{thread_id}/messages",
    response_model=ValidationInterface.MessagesList,
)
def list_messages(
    thread_id: str,
    limit: int = 20,
    order: str = "asc",
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Listing messages for thread {thread_id}"
    )
    svc = MessageService()
    try:
        return svc.list_messages(
            thread_id=thread_id, user_id=auth_key.user_id, limit=limit, order=order
        )
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error("Error listing messages: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get(
    "/threads/{thread_id}/formatted_messages", response_model=List[Dict[str, Any]]
)
def get_formatted_messages(
    thread_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Retrieving formatted messages for thread {thread_id}"
    )
    svc = MessageService()
    try:
        return svc.get_formatted_messages(thread_id, user_id=auth_key.user_id)
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(
            f"Error retrieving formatted messages: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.post("/messages/assistant", response_model=ValidationInterface.MessageRead)
async def save_assistant_message(
    message: ValidationInterface.MessageCreate,
    redis: Redis = Depends(get_redis),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(
        f"[{auth_key.user_id}] Saving assistant chunk for thread {message.thread_id}"
    )
    # NOTE: save_assistant_message_chunk is an internal/trusted-caller path used
    # by the execution engine — it does not perform end-user ownership checks.
    # The endpoint still requires a valid API key for authentication.
    svc = MessageService()
    try:
        new_message = svc.save_assistant_message_chunk(
            thread_id=message.thread_id,
            content=message.content,
            role=message.role,
            assistant_id=message.assistant_id,
            sender_id=message.sender_id,
            is_last_chunk=True,
        )
        if new_message:
            logging_utility.info(f"Saved assistant message ID {new_message.id}")
            await _push_to_redis(redis, new_message)
            return new_message
        return None
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Error saving assistant message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.delete(
    "/messages/{message_id}",
    response_model=ValidationInterface.MessageDeleted,
)
def delete_message(
    message_id: str,
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    logging_utility.info(f"[{auth_key.user_id}] Deleting message {message_id}")
    svc = MessageService()
    try:
        return svc.delete_message(message_id, user_id=auth_key.user_id)
    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error("Error deleting message: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )
