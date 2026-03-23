import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from redis import Redis

from src.api.entities_api.dependencies import get_redis
from src.api.entities_api.orchestration.engine.inference_arbiter import InferenceArbiter
from src.api.entities_api.orchestration.engine.inference_provider_selector import (
    InferenceProviderSelector,
)
from src.api.entities_api.services.native_execution_service import NativeExecutionService

router = APIRouter()
logging_utility = LoggingUtility()


@router.post(
    "/completions",
    summary="Completions endpoint — streaming (default) or buffered",
    response_description="SSE stream or single JSON object depending on stream flag",
)
async def completions(
    stream_request: ValidationInterface.StreamRequest,
    redis: Redis = Depends(get_redis),
):
    logging_utility.info(
        "Completions endpoint called — model: %s, run: %s, stream: %s",
        stream_request.model,
        stream_request.run_id,
        stream_request.stream,
    )

    # ------------------------------------------------------------------
    # OWNERSHIP GUARD
    #
    # The run is the trust anchor. It was created under auth, so its
    # user_id is reliable. We verify that the thread and assistant IDs
    # in the request are consistent with the run — this closes the
    # mismatched-ID attack vector without requiring a second API key.
    #
    # ARCH NOTE: This endpoint requires two separate credentials:
    #   - stream_request.api_key → Inference provider key (forwarded to LLM)
    #   - Project David identity  → Derived from run.user_id (trust anchor)
    #
    # We cannot add get_api_key here without breaking the dual-key
    # constraint: the client cannot simultaneously pass a Project David
    # key in the header AND an inference provider key in the body through
    # the current SDK flow. Identity is therefore established via the run.
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
        )

    except HTTPException:
        raise
    except Exception as e:
        logging_utility.error(f"Ownership check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ownership verification failed.")

    # ------------------------------------------------------------------
    # PROVIDER SETUP
    # ------------------------------------------------------------------
    try:
        arbiter = InferenceArbiter(redis=redis)
        selector = InferenceProviderSelector(arbiter)
        general_handler_instance, api_model_name = selector.select_provider(
            model_id=stream_request.model
        )
    except Exception as e:
        logging_utility.error(f"Provider setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
            model=stream_request.model,
            stream_reasoning=False,
            api_key=stream_request.api_key,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # PATH A: STREAMING (default)
    #
    # Each chunk is forwarded to the client as it arrives via SSE.
    # ------------------------------------------------------------------
    if stream_request.stream:

        async def stream_generator():
            start_time = time.time()
            run_id = stream_request.run_id
            prefix = "data: "
            suffix = "\n\n"
            chunk_count = 0
            error_occurred = False

            try:
                async for chunk in event_generator():
                    chunk_count += 1
                    final_str = ""

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

                    yield f"{prefix}{final_str}{suffix}"

                if not error_occurred:
                    yield "data: [DONE]\n\n"

            except Exception as e:
                error_occurred = True
                logging_utility.error(f"Stream loop error: {e}", exc_info=True)
                yield (
                    f"data: {json.dumps({'type': 'error', 'run_id': run_id, 'message': str(e)})}\n\n"
                )
            finally:
                elapsed = time.time() - start_time
                logging_utility.info(f"Stream finished: {chunk_count} chunks in {elapsed:.2f}s")

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
            # Normalise to dict
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
                # Assemble the actual response text
                content_parts.append(parsed.get("content", ""))

            elif chunk_type == "error":
                # Capture the first error and abort assembly
                error_message = parsed.get("message") or parsed.get("error", "Unknown error")
                logging_utility.error(
                    f"[{run_id}] Buffered mode — error chunk received: {error_message}"
                )
                break

            else:
                # Side-effect events (tool_call_start, research_status, web_status,
                # code_status, shell_status, hot_code, hot_code_output,
                # computer_output, code_interpreter_file, computer_file,
                # scratchpad_status, engineer_status, reasoning) —
                # all execute normally, logged here for observability.
                logging_utility.info(f"[{run_id}] Buffered side-effect: type={chunk_type}")

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
            "model": stream_request.model,
            "elapsed_s": round(elapsed, 3),
        }
    )
