import json
import time
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from redis import Redis

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.dependencies import get_api_key, get_redis
from src.api.entities_api.models.models import Run
from src.api.entities_api.orchestration.engine.inference_arbiter import InferenceArbiter
from src.api.entities_api.orchestration.engine.inference_provider_selector import (
    InferenceProviderSelector,
)
from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
)
from src.api.entities_api.services.native_execution_service import (
    NativeExecutionService,
)
from src.api.entities_api.services.provider_key_service import (
    ProviderKeyConfigurationError,
    ProviderKeyNotFoundError,
    ProviderKeyUnavailableError,
)
from src.api.entities_api.services.stateless_inference_service import (
    StatelessInferenceService,
)

router = APIRouter()
logging_utility = LoggingUtility()


class StatelessCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class StatelessCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = Field(min_length=1, max_length=500)
    prompt: Optional[str] = Field(default=None, min_length=1, max_length=16_000)
    messages: Optional[List[StatelessCompletionMessage]] = Field(
        default=None,
        min_length=1,
        max_length=32,
    )
    api_key: Optional[str] = None
    provider_key_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: Literal[False] = False
    stateless: Literal[True]

    @field_validator("model", "prompt")
    @classmethod
    def strings_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def exactly_one_input_shape(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        return self

    def inference_messages(self) -> List[Dict[str, str]]:
        if self.messages is not None:
            return [message.model_dump() for message in self.messages]
        return [{"role": "user", "content": self.prompt or ""}]


class StatefulCompletionRequest(ValidationInterface.StreamRequest):
    """Stateful inference contract with an explicit provider key reference."""

    stateless: Literal[False] = False
    provider_key_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


CompletionRequest = StatelessCompletionRequest | StatefulCompletionRequest


async def _authenticate_stateless_request(api_key_header: Optional[str]) -> str:
    """Authenticate stateless calls and return the trusted owner identity."""
    with SessionLocal() as db:
        authenticated_key = await get_api_key(api_key_header=api_key_header, db=db)
        return authenticated_key.user_id


async def _resolve_provider_api_key(
    *,
    model: str,
    user_id: str,
    provider_key_id: Optional[str],
    raw_api_key: Optional[str],
) -> Optional[str]:
    """Gate Hyperbolic BYOK before dispatch; leave other providers unchanged.

    Registered credentials never enter cache, model metadata, or HTTP responses.
    The raw internal key remains local to the current inference invocation.
    """
    if not model.strip().lower().startswith("hyperbolic/"):
        if provider_key_id is not None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Registered provider key selection is not enabled "
                    "for this model provider."
                ),
            )
        return raw_api_key  # Legacy providers; migrated separately.

    if raw_api_key is not None:
        raise HTTPException(
            status_code=422,
            detail="Direct Hyperbolic API keys are not accepted. Use provider_key_id.",
        )
    if not provider_key_id:
        raise HTTPException(
            status_code=422,
            detail="A registered provider_key_id is required for Hyperbolic inference.",
        )

    try:
        return await NativeExecutionService.resolve_provider_key_for_inference(
            provider_key_id=provider_key_id,
            user_id=user_id,
            provider="hyperbolic",
        )
    except (ProviderKeyNotFoundError, ProviderKeyUnavailableError) as exc:
        # Do not disclose whether a foreign, revoked, or mismatched key exists.
        raise HTTPException(
            status_code=404, detail="Provider key unavailable."
        ) from exc
    except (ProviderKeyConfigurationError, CredentialConfigurationError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Provider credential security configuration unavailable.",
        ) from exc


# ── Rust SSE framer (Phase 2 hot-path) ───────────────────────────────────────
# frame_sse_chunk handles type dispatch, run_id injection, JSON serialisation,
# and SSE framing ("data: ...\n\n") entirely in Rust outside the GIL.
# Falls back transparently to the pure-Python implementation if unavailable.
try:
    from delta_normalizer_rs import frame_sse_chunk as _rs_frame

    def _frame_chunk(chunk, run_id: str) -> str:
        return _rs_frame(chunk, run_id)

    logging_utility.info(
        "inference_router: Rust SSE framer active (delta_normalizer_rs)"
    )

except ImportError:
    logging_utility.warning(
        "inference_router: Rust SSE framer not available — using pure Python fallback. "
        "Build with: cd rust/delta_normalizer && maturin develop --release"
    )

    def _frame_chunk(chunk, run_id: str) -> str:  # type: ignore[misc]
        if isinstance(chunk, str):
            final_str = chunk
        elif isinstance(chunk, dict):
            if "run_id" not in chunk:
                chunk["run_id"] = run_id
            final_str = json.dumps(chunk)
        else:
            final_str = json.dumps(
                {"type": "content", "content": str(chunk), "run_id": run_id}
            )
        return f"data: {final_str}\n\n"


@router.post(
    "/completions",
    summary="Completions endpoint — streaming (default) or buffered",
    response_description="SSE stream or single JSON object depending on stream flag",
)
async def completions(
    stream_request: CompletionRequest,
    redis: Redis = Depends(get_redis),
    project_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    if isinstance(stream_request, StatelessCompletionRequest):
        authenticated_user_id = await _authenticate_stateless_request(project_api_key)
        provider_api_key = await _resolve_provider_api_key(
            model=stream_request.model,
            user_id=authenticated_user_id,
            provider_key_id=stream_request.provider_key_id,
            raw_api_key=stream_request.api_key,
        )
        logging_utility.info(
            "Stateless completion called — model: %s, stream: false",
            stream_request.model,
        )
        try:
            completion = await StatelessInferenceService(redis=redis).create_completion(
                model=stream_request.model,
                messages=stream_request.inference_messages(),
                provider_api_key=provider_api_key,
                max_tokens=stream_request.max_tokens,
                temperature=stream_request.temperature,
                top_p=stream_request.top_p,
            )
        except Exception as exc:
            logging_utility.error(
                "Stateless completion failed: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Stateless completion failed: {exc}",
            ) from exc

        return JSONResponse(
            content={
                "object": "text_completion",
                "created": int(time.time()),
                "model": stream_request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": completion,
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    logging_utility.info(
        "Completions endpoint called — model: %s, run: %s, stream: %s",
        stream_request.model,
        stream_request.run_id,
        stream_request.stream,
    )

    # ------------------------------------------------------------------
    # OWNERSHIP GUARD
    #
    # The Run binds the thread, assistant, and owner. The caller must
    # authenticate as that owner before execution is permitted.
    #
    # Credentials stay separate:
    #   - stream_request.api_key → Inference provider key (JSON body)
    #   - project_api_key        → Project David key (X-API-Key header)
    #
    # Chain of trust:
    #   1. Run exists and was created under auth (run.user_id is trusted)
    #   2. run.thread_id matches the request (no thread grafting)
    #   3. run.assistant_id matches the request (no assistant grafting)
    #   4. Caller has owner/shared access to the assistant
    # ------------------------------------------------------------------
    try:
        native = NativeExecutionService()

        run = await native.retrieve_run(stream_request.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")

        # Authenticate the Project David X-API-Key and bind it to the run owner
        # before allowing the credential into delegated execution.
        with SessionLocal() as db:
            authenticated_key = await get_api_key(
                api_key_header=project_api_key,
                db=db,
            )
            authenticated_user_id = authenticated_key.user_id
            authenticated_is_protected = bool(authenticated_key.is_protected)

        if authenticated_user_id != run.user_id:
            raise HTTPException(
                status_code=403,
                detail="API key identity does not match the run owner.",
            )

        if run.thread_id != stream_request.thread_id:
            raise HTTPException(
                status_code=403,
                detail="Thread ID does not match the run.",
            )

        if run.assistant_id != stream_request.assistant_id:
            raise HTTPException(
                status_code=403,
                detail="Assistant ID does not match the run.",
            )

        await native.assert_assistant_access(
            assistant_id=stream_request.assistant_id,
            user_id=run.user_id,
            is_protected_key=authenticated_is_protected,
        )

        # The inference request selects the execution model; Run.model may
        # still contain the Assistant default until the first inference.
        effective_model = stream_request.model.strip()
        if not effective_model:
            raise HTTPException(status_code=422, detail="Inference model is required.")

    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Ownership check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ownership verification failed.")

    # Resolve credentials only AFTER authentication and Run ownership checks.
    # Do this before dispatch or durable Run model mutation, so credential
    # rejection is a normal HTTP error rather than a leaked SSE error chunk.
    provider_api_key = await _resolve_provider_api_key(
        model=effective_model,
        user_id=authenticated_user_id,
        provider_key_id=stream_request.provider_key_id,
        raw_api_key=stream_request.api_key,
    )

    # ------------------------------------------------------------------
    # PROVIDER SETUP
    # ------------------------------------------------------------------
    try:
        arbiter = InferenceArbiter(redis=redis)
        selector = InferenceProviderSelector(arbiter)
        general_handler_instance, api_model_name = selector.select_provider(
            model_id=effective_model
        )
    except ValueError as e:
        logging_utility.warning("Invalid inference model for provider routing: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logging_utility.error(f"Provider setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Bind the validated inference model to the durable Run for accounting.
    # Only a queued Run may adopt a different model. A row lock prevents
    # competing writes, and the ownership/identity constraints are rechecked.
    with SessionLocal() as db:
        persisted_run = (
            db.query(Run)
            .filter(
                Run.id == stream_request.run_id,
                Run.user_id == authenticated_user_id,
                Run.thread_id == stream_request.thread_id,
                Run.assistant_id == stream_request.assistant_id,
            )
            .with_for_update()
            .first()
        )
        if persisted_run is None:
            raise HTTPException(status_code=404, detail="Run not found.")

        run_status = getattr(persisted_run.status, "value", persisted_run.status)
        if persisted_run.model != effective_model:
            if run_status != "queued":
                raise HTTPException(
                    status_code=409,
                    detail="Cannot change the Run model after execution begins.",
                )
            persisted_run.model = effective_model
            db.commit()

    # ------------------------------------------------------------------
    # CORE GENERATOR
    #
    # Runs identically regardless of stream mode. All side effects —
    # tool calls, file generation, status events — execute in both paths.
    # The stream flag only controls how the output is delivered to the
    # caller, not what the generator does internally.
    # ------------------------------------------------------------------
    async def event_generator():
        run_id = stream_request.run_id
        async for chunk in general_handler_instance.process_conversation(
            thread_id=stream_request.thread_id,
            message_id=stream_request.message_id,
            run_id=run_id,
            assistant_id=stream_request.assistant_id,
            model=effective_model,
            stream_reasoning=False,
            api_key=provider_api_key,
            project_david_api_key=project_api_key,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # PATH A: STREAMING (default)
    #
    # Each chunk is forwarded to the client as it arrives via SSE.
    # Rust handles type dispatch, run_id injection, JSON serialisation,
    # and frame construction ("data: ...\n\n") outside the GIL.
    # ------------------------------------------------------------------
    if stream_request.stream:

        async def stream_generator():
            start_time = time.time()
            run_id = stream_request.run_id
            chunk_count = 0
            error_occurred = False

            try:
                async for chunk in event_generator():
                    chunk_count += 1
                    yield _frame_chunk(chunk, run_id)

                if not error_occurred:
                    yield "data: [DONE]\n\n"

            except Exception as e:
                error_occurred = True
                logging_utility.error(f"Stream loop error: {e}", exc_info=True)
                yield _frame_chunk(
                    {"type": "error", "run_id": run_id, "message": str(e)},
                    run_id,
                )
            finally:
                elapsed = time.time() - start_time
                logging_utility.info(
                    f"Stream finished: {chunk_count} chunks in {elapsed:.2f}s"
                )

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    # ------------------------------------------------------------------
    # PATH B: BUFFERED
    #
    # The generator runs to completion server-side. Content chunks are
    # assembled into a single string. All other event types (tool calls,
    # status events, file events) still execute — they are logged but
    # not forwarded since the client is waiting for a single response.
    #
    # Response shape mirrors a non-streaming OpenAI-compatible completion:
    # {
    #   "run_id":    "<run_id>",
    #   "content":   "<assembled text>",
    #   "type":      "content",
    #   "model":     "<model>",
    #   "elapsed_s": <float>
    # }
    # ------------------------------------------------------------------
    start_time = time.time()
    run_id = stream_request.run_id
    content_parts = []
    error_message = None

    try:
        async for chunk in event_generator():
            if isinstance(chunk, str):
                try:
                    parsed = json.loads(chunk)
                except Exception:
                    parsed = {"type": "content", "content": chunk}
            elif isinstance(chunk, dict):
                parsed = chunk
            else:
                parsed = {"type": "content", "content": str(chunk)}

            chunk_type = parsed.get("type", "content")

            if chunk_type == "content":
                content_parts.append(parsed.get("content", ""))

            elif chunk_type == "error":
                error_message = parsed.get("message") or parsed.get(
                    "error", "Unknown error"
                )
                logging_utility.error(
                    f"[{run_id}] Buffered mode — error chunk received: {error_message}"
                )
                break

            else:
                logging_utility.info(
                    f"[{run_id}] Buffered side-effect: type={chunk_type}"
                )

    except Exception as e:
        logging_utility.error(f"Buffered generator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Buffered completion failed: {e}")

    elapsed = time.time() - start_time

    if error_message:
        raise HTTPException(status_code=500, detail=error_message)

    assembled_content = "".join(content_parts)
    logging_utility.info(
        f"[{run_id}] Buffered response assembled: "
        f"{len(assembled_content)} chars in {elapsed:.2f}s"
    )

    return JSONResponse(
        content={
            "run_id": run_id,
            "content": assembled_content,
            "type": "content",
            "model": effective_model,
            "elapsed_s": round(elapsed, 3),
        }
    )
