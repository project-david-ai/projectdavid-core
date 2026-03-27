# src/api/entities_api/services/message_service.py
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Message, Thread
from src.api.entities_api.services.logging_service import LoggingUtility

validator = ValidationInterface()
logging_utility = LoggingUtility()


class MessageService:

    def __init__(self):
        self.message_chunks: Dict[str, List[str]] = {}
        logging_utility.info(f"Initialized MessageService. Source: {__file__}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_for_read(self, db_msg: Message) -> Message:
        """Ensure meta_data is a dict before Pydantic validation."""
        if isinstance(db_msg.meta_data, str):
            try:
                db_msg.meta_data = json.loads(db_msg.meta_data)
            except (TypeError, ValueError, json.JSONDecodeError):
                db_msg.meta_data = {}
        elif db_msg.meta_data is None:
            db_msg.meta_data = {}
        return db_msg

    def _assert_thread_owner(self, db: Any, thread_id: str, user_id: str) -> "Thread":
        db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not db_thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        if db_thread.owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this thread.",
            )
        return db_thread

    def _assert_message_owner(
        self, db: Any, message_id: str, user_id: str
    ) -> "Message":
        db_message = db.query(Message).filter(Message.id == message_id).first()
        if not db_message:
            raise HTTPException(status_code=404, detail="Message not found")
        db_thread = db.query(Thread).filter(Thread.id == db_message.thread_id).first()
        if not db_thread or db_thread.owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this message.",
            )
        return db_message

    def _hydrate_attachments(
        self,
        db: Any,
        text_content: str,
        attachments: List[Dict[str, Any]],
    ) -> Any:
        """
        Resolve file_ids → base64 and return a Qwen multimodal content array:

            [
                {"type": "text",  "text":  "<original message text>"},
                {"type": "image", "image": "data:image/jpeg;base64,..."},
                ...
            ]

        Expired / deleted files are silently skipped.
        If every attachment has expired the original plain text string is
        returned so the message still reaches the model.

        NOTE: called ONLY at LLM consumption time — never when populating Redis.
        """
        from src.api.entities_api.services.file_service import FileService

        image_attachments = [a for a in (attachments or []) if a.get("type") == "image"]
        if not image_attachments:
            return text_content

        file_svc = FileService(db)
        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]

        resolved = 0
        for attachment in image_attachments:
            file_id = attachment.get("file_id")
            if not file_id:
                continue

            b64 = file_svc.get_file_as_base64_internal(file_id)
            if b64 is None:
                logging_utility.warning(
                    "_hydrate_attachments: file_id=%s not found or expired, skipping.",
                    file_id,
                )
                continue

            mime = _detect_mime_from_b64(b64)
            content_blocks.append(
                {"type": "image", "image": f"data:{mime};base64,{b64}"}
            )
            resolved += 1

        if resolved == 0:
            logging_utility.warning(
                "_hydrate_attachments: all attachments expired, returning plain text."
            )
            return text_content

        return content_blocks

    def _format_messages_from_db(
        self,
        messages: List[Message],
        hydrate_images: bool = False,
        include_attachments: bool = False,
        db: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Shared formatting loop used by all three paths:

        hydrate_images=False, include_attachments=False
            → plain text content, no attachment metadata
            → used by legacy callers that don't need image support

        hydrate_images=False, include_attachments=True
            → plain text content + attachments:[{type, file_id}] preserved
            → CACHE PATH — Redis stores lean file_id refs, never base64

        hydrate_images=True, include_attachments=False
            → file_ids resolved to base64 Qwen content arrays
            → LLM PATH — called at LLM consumption time only
            → db must be provided
        """
        formatted_messages: List[Dict[str, Any]] = []

        for db_message in messages:
            role = db_message.role

            if role == "tool":
                formatted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": db_message.tool_call_id,
                        "content": db_message.content,
                    }
                )
                continue

            if role == "assistant":
                try:
                    parsed = json.loads(db_message.content)
                    is_tool_list = (
                        isinstance(parsed, list)
                        and len(parsed) > 0
                        and all(isinstance(i, dict) and "function" in i for i in parsed)
                    )
                    if is_tool_list:
                        formatted_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": parsed,
                            }
                        )
                    else:
                        formatted_messages.append(
                            {"role": "assistant", "content": db_message.content}
                        )
                except (json.JSONDecodeError, TypeError):
                    formatted_messages.append(
                        {"role": "assistant", "content": db_message.content}
                    )
                continue

            # ── User / system / platform messages ─────────────────────────
            attachments = db_message.attachments or []

            if hydrate_images and attachments and db is not None:
                # LLM path: resolve file_ids → base64 content array
                content = self._hydrate_attachments(db, db_message.content, attachments)
                formatted_messages.append({"role": role, "content": content})

            elif include_attachments and attachments:
                # Cache path: keep plain text + preserve attachment metadata
                # so the mixin can hydrate just-in-time before LLM dispatch
                formatted_messages.append(
                    {
                        "role": role,
                        "content": db_message.content,
                        "attachments": attachments,  # ← lean file_id refs only
                    }
                )

            else:
                # Plain text — no attachments or not needed
                formatted_messages.append({"role": role, "content": db_message.content})

        return formatted_messages

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal / trusted-caller variants  (NO ownership check)
    # ──────────────────────────────────────────────────────────────────────────

    def create_message_internal(
        self,
        message: validator.MessageCreate,
    ) -> validator.MessageRead:
        """FOR INTERNAL/TRUSTED CALLERS ONLY."""
        logging_utility.info(
            f"[INTERNAL] Creating message for thread_id={message.thread_id}, role={message.role}."
        )
        with SessionLocal() as db:
            db_thread = db.query(Thread).filter(Thread.id == message.thread_id).first()
            if not db_thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            db_message = Message(
                id=UtilsInterface.IdentifierService.generate_message_id(),
                assistant_id=message.assistant_id,
                attachments=message.attachments or [],
                completed_at=None,
                content=message.content,
                created_at=int(time.time()),
                incomplete_at=None,
                incomplete_details=None,
                meta_data=message.meta_data or {},
                object="message",
                role=message.role,
                run_id=None,
                status=None,
                thread_id=message.thread_id,
                sender_id=message.sender_id,
                tool_id=getattr(message, "tool_id", None),
                tool_call_id=getattr(message, "tool_call_id", None),
            )
            try:
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
            except Exception as e:
                db.rollback()
                logging_utility.error(f"[INTERNAL] Error saving message: {e}")
                raise HTTPException(status_code=500, detail="Failed to create message")

            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    def get_raw_messages_internal(
        self,
        thread_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Return lean formatted messages — plain text content with attachment
        metadata (file_id refs) preserved as a sibling key.

        FOR CACHE USE ONLY — MessageCache cold-load path.

        Output shape for messages with images:
            {"role": "user", "content": "...", "attachments": [{"type": "image", "file_id": "..."}]}

        Output shape for plain messages:
            {"role": "user", "content": "..."}

        Redis stores these lean dicts. The mixin hydrates images just before
        LLM dispatch — base64 bytes are never written to Redis.
        """
        with SessionLocal() as db:
            db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
            if not db_thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            messages = (
                db.query(Message)
                .filter(Message.thread_id == thread_id)
                .order_by(Message.created_at.asc())
                .all()
            )

            return self._format_messages_from_db(
                messages,
                hydrate_images=False,
                include_attachments=True,  # ← preserve file_id refs for mixin
            )

    def get_formatted_messages_internal(
        self,
        thread_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Return fully hydrated messages — image attachments resolved to base64.

        FOR LLM CONSUMPTION ONLY — NativeExecutionService / orchestrator.
        DO NOT use this to populate Redis.
        """
        with SessionLocal() as db:
            db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
            if not db_thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            messages = (
                db.query(Message)
                .filter(Message.thread_id == thread_id)
                .order_by(Message.created_at.asc())
                .all()
            )

            return self._format_messages_from_db(
                messages,
                hydrate_images=True,
                include_attachments=False,
                db=db,
            )

    def submit_tool_output_internal(
        self,
        message: validator.MessageCreate,
    ) -> validator.MessageRead:
        """FOR INTERNAL/TRUSTED CALLERS ONLY."""
        with SessionLocal() as db:
            db_thread = db.query(Thread).filter(Thread.id == message.thread_id).first()
            if not db_thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            db_message = Message(
                id=UtilsInterface.IdentifierService.generate_message_id(),
                assistant_id=message.assistant_id,
                content=message.content,
                created_at=int(time.time()),
                meta_data=message.meta_data or {},
                object="message",
                role="tool",
                thread_id=message.thread_id,
                tool_id=message.tool_id,
                tool_call_id=message.tool_call_id,
            )
            try:
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail="Failed to create message")

            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API  (ownership enforced)
    # ──────────────────────────────────────────────────────────────────────────

    def create_message(
        self,
        message: validator.MessageCreate,
        user_id: str,
    ) -> validator.MessageRead:
        logging_utility.info(
            f"Creating message for thread_id={message.thread_id}, role={message.role}."
        )
        with SessionLocal() as db:
            self._assert_thread_owner(db, message.thread_id, user_id)

            db_message = Message(
                id=UtilsInterface.IdentifierService.generate_message_id(),
                assistant_id=message.assistant_id,
                attachments=message.attachments or [],
                completed_at=None,
                content=message.content,
                created_at=int(time.time()),
                incomplete_at=None,
                incomplete_details=None,
                meta_data=message.meta_data or {},
                object="message",
                role=message.role,
                run_id=None,
                status=None,
                thread_id=message.thread_id,
                sender_id=message.sender_id,
                tool_id=getattr(message, "tool_id", None),
                tool_call_id=getattr(message, "tool_call_id", None),
            )
            try:
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
            except Exception as e:
                db.rollback()
                logging_utility.error(f"Error saving message: {e}")
                raise HTTPException(status_code=500, detail="Failed to create message")

            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    def retrieve_message(self, message_id: str, user_id: str) -> validator.MessageRead:
        with SessionLocal() as db:
            db_message = self._assert_message_owner(db, message_id, user_id)
            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    def list_messages(
        self,
        thread_id: str,
        user_id: str,
        limit: int = 20,
        order: str = "asc",
    ) -> validator.MessagesList:
        with SessionLocal() as db:
            self._assert_thread_owner(db, thread_id, user_id)
            query = db.query(Message).filter(Message.thread_id == thread_id)
            query = (
                query.order_by(Message.created_at.asc())
                if order == "asc"
                else query.order_by(Message.created_at.desc())
            )
            db_messages = query.limit(limit).all()
            messages = [
                validator.MessageRead.model_validate(self._prepare_for_read(m))
                for m in db_messages
            ]
            return validator.MessagesList(
                data=messages,
                first_id=messages[0].id if messages else None,
                last_id=messages[-1].id if messages else None,
                has_more=len(db_messages) >= limit,
            )

    def save_assistant_message_chunk(
        self,
        thread_id: str,
        content: str,
        role: str,
        assistant_id: str,
        sender_id: str,
        is_last_chunk: bool = False,
    ) -> Optional[validator.MessageRead]:
        if thread_id not in self.message_chunks:
            self.message_chunks[thread_id] = []
        self.message_chunks[thread_id].append(content)

        if not is_last_chunk:
            return None

        complete_message = "".join(self.message_chunks.pop(thread_id, []))

        with SessionLocal() as db:
            db_message = Message(
                id=UtilsInterface.IdentifierService.generate_message_id(),
                assistant_id=assistant_id,
                attachments=[],
                completed_at=int(time.time()),
                content=complete_message,
                created_at=int(time.time()),
                meta_data={},
                object="message",
                role=role,
                thread_id=thread_id,
                sender_id=sender_id,
            )
            try:
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail="Failed to save message")

            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    def get_formatted_messages(
        self,
        thread_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Public API — ownership enforced. Hydrates images for LLM consumption."""
        with SessionLocal() as db:
            self._assert_thread_owner(db, thread_id, user_id)
            messages = (
                db.query(Message)
                .filter(Message.thread_id == thread_id)
                .order_by(Message.created_at.asc())
                .all()
            )
            return self._format_messages_from_db(
                messages,
                hydrate_images=True,
                include_attachments=False,
                db=db,
            )

    def submit_tool_output(
        self,
        message: validator.MessageCreate,
        user_id: str,
    ) -> validator.MessageRead:
        with SessionLocal() as db:
            self._assert_thread_owner(db, message.thread_id, user_id)
            db_message = Message(
                id=UtilsInterface.IdentifierService.generate_message_id(),
                assistant_id=message.assistant_id,
                content=message.content,
                created_at=int(time.time()),
                meta_data=message.meta_data or {},
                object="message",
                role="tool",
                thread_id=message.thread_id,
                tool_id=message.tool_id,
                tool_call_id=message.tool_call_id,
            )
            try:
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail="Failed to create message")

            return validator.MessageRead.model_validate(
                self._prepare_for_read(db_message)
            )

    def delete_message(self, message_id: str, user_id: str) -> validator.MessageDeleted:
        with SessionLocal() as db:
            self._assert_message_owner(db, message_id, user_id)
            db_msg = db.query(Message).filter(Message.id == message_id).first()
            db.delete(db_msg)
            db.commit()
            return validator.MessageDeleted(id=message_id)


# ──────────────────────────────────────────────────────────────────────────────
#  Module-level helper
# ──────────────────────────────────────────────────────────────────────────────


def _detect_mime_from_b64(b64_string: str) -> str:
    """Peek at first decoded bytes to detect image MIME type."""
    import base64 as _base64

    try:
        header = _base64.b64decode(b64_string[:16])
        if header[:2] == b"\xff\xd8":
            return "image/jpeg"
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"
