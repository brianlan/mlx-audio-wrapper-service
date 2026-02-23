import asyncio
import json
import time
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import mlx.core as mx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel
from mlx_audio.stt.utils import load_model
from starlette.requests import Request
from starlette.responses import Response

from asr_service.config import config
from asr_service.observability import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    add_request_id_middleware,
    get_metrics,
    log_request,
    track_duration,
    track_request,
)
from asr_service.validators import (
    DurationError,
    FileSizeError,
    FormatError,
    TimeoutError,
    TimeoutContext,
    validate_input,
)
from asr_service.queue import QueuedRequest, get_request_queue


MODEL_PATH = config.model_path

app = FastAPI(title="MLX Qwen3-ASR Service", version="1.0")

# Add request ID middleware
app.middleware("http")(add_request_id_middleware)

state = {
    "model": None,
    "device": None,
    "ready": False,
    "warmup_error": None,
    "shutting_down": False,
    "stream_lock": asyncio.Lock(),
}


async def _process_queued_request(queued_request: QueuedRequest) -> dict:
    """Process a queued transcription request.

    This function runs in the background worker and handles the actual
    model inference for requests that have been queued.

    Args:
        queued_request: The queued request to process

    Returns:
        Dict with transcription result (same format as endpoint response)
    """
    from asr_service.observability import logger as obs_logger

    temp_path = None
    try:
        # Write file data to temp file
        suffix = queued_request.file_suffix or ".wav"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(queued_request.file_data)

        forced_language = _normalize_language(queued_request.language)

        # Use timeout context for request timeout protection
        with TimeoutContext(config.request_timeout_sec) as timeout_ctx:
            timeout_ctx.check_timeout()

            output = generate_transcription(
                model=state["model"],
                audio=str(temp_path),
                language=forced_language,
                verbose=False,
            )

            timeout_ctx.check_timeout()

        detected_language, cleaned_text = _parse_asr_output(
            output.text,
            forced_language=forced_language,
        )

        return {
            "text": cleaned_text,
            "language": detected_language,
            "segments": output.segments,
            "usage": {
                "prompt_tokens": output.prompt_tokens,
                "generation_tokens": output.generation_tokens,
                "total_tokens": output.total_tokens,
            },
            "runtime": {
                "mlx_default_device": state.get("device", "unknown"),
                "total_time_s": output.total_time,
                "prompt_tps": output.prompt_tps,
                "generation_tps": output.generation_tps,
            },
            "input": {
                "language": queued_request.language,
                "mode": "auto" if forced_language is None else "forced",
            },
        }
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"auto", "none", "null"}:
        return None
    return normalized


def _parse_asr_output(raw: str, forced_language: Optional[str]) -> tuple[Optional[str], str]:
    s = (raw or "").strip()
    if not s:
        return forced_language, ""

    if "<asr_text>" not in s:
        return forced_language, s

    meta, text = s.split("<asr_text>", 1)
    text = text.strip()

    if forced_language is not None:
        return forced_language, text

    detected_language: Optional[str] = None
    for line in meta.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("language "):
            candidate = line[len("language ") :].strip()
            if candidate and candidate.lower() != "none":
                detected_language = candidate
            break

    return detected_language, text


def _patch_qwen3_prompt_for_auto_lid() -> None:
    if state.get("auto_lid_patch_applied"):
        return

    original = Qwen3ASRModel._build_prompt

    def patched(self, num_audio_tokens: int, language: str = "English"):
        if language is None:
            prompt = (
                f"<|im_start|>system\n<|im_end|>\n"
                f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            input_ids = self._tokenizer.encode(prompt, return_tensors="np")
            return mx.array(input_ids)
        return original(self, num_audio_tokens, language)

    Qwen3ASRModel._build_prompt = patched
    state["auto_lid_patch_applied"] = True


def _warmup_model() -> None:
    """Run a warmup inference to ensure model is fully loaded and ready."""
    import io
    import numpy as np
    import wave

    # Create a minimal silent audio buffer (1 second of silence at 16kHz)
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    audio_data = np.zeros(num_samples, dtype=np.float32)

    # Write to temporary WAV file
    with io.BytesIO() as buffer:
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        buffer.seek(0)

        # Save to temp file for mlx_audio
        with NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(buffer.read())
            temp_path = Path(tmp.name)

    try:
        # Run a quick inference to warm up
        output = generate_transcription(
            model=state["model"],
            audio=str(temp_path),
            language=None,  # Use auto-detection
            verbose=False,
        )
        state["ready"] = True
    except Exception as e:
        state["warmup_error"] = str(e)
        state["ready"] = False
        raise RuntimeError(f"Warmup failed: {e}") from e
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.on_event("startup")
async def startup_event() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")
    _patch_qwen3_prompt_for_auto_lid()
    state["model"] = load_model(str(MODEL_PATH))
    state["device"] = str(mx.default_device())
    # Run warmup inference to ensure model is ready
    _warmup_model()
    # Start the request queue worker
    queue = get_request_queue()
    await queue.start_worker(_process_queued_request)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Graceful shutdown handler - stop accepting new requests and drain queue."""
    state["shutting_down"] = True

    # Stop accepting new requests (queue will reject with 429)
    queue = get_request_queue()
    await queue.stop_worker(timeout=config.request_timeout_sec)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "mlx_default_device": state.get("device", "unknown"),
    }


@app.get("/live")
def liveness() -> dict:
    """Liveness probe - returns 200 if process is running."""
    return {"status": "alive"}


@app.get("/ready")
def readiness() -> JSONResponse:
    """Readiness probe - returns 200 if model is ready to serve requests."""
    if not state.get("ready"):
        # Not ready - return 503 with Retry-After
        detail = {
            "status": "not_ready",
            "reason": state.get("warmup_error", "warmup in progress or failed"),
        }
        if state.get("warmup_error"):
            detail["error"] = state["warmup_error"]
        
        return JSONResponse(
            status_code=503,
            content=detail,
            headers={"Retry-After": "30"},
        )

    # Check if queue is full
    queue = get_request_queue()
    queue_info = {
        "current_size": queue.current_size,
        "max_size": queue.max_size,
        "is_full": queue.is_full,
    }

    # If queue is full, still return 200 but include warning in response
    # (better than 503 since service is technically ready, just busy)
    response_content = {
        "status": "ready",
        "model_path": str(MODEL_PATH),
        "device": state.get("device", "unknown"),
        "queue": queue_info,
    }

    if queue.is_full:
        response_content["warning"] = "queue_full"

    return JSONResponse(status_code=200, content=response_content)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type="text/plain")


@app.post("/v1/audio/transcriptions")
@track_duration("POST", "/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=str(MODEL_PATH)),
    language: str = Form(default="auto"),
    stream: bool = Form(default=False),
    response_format: str = Form(default="json"),
    include_accumulated: bool = Form(default=False),
    request: Request = None,  # type: ignore
) -> dict:
    # Check if service is shutting down
    if state.get("shutting_down"):
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/v1/audio/transcriptions",
            status="service_unavailable",
        ).inc()
        raise HTTPException(
            status_code=503,
            detail="Service is shutting down, please retry later",
        )

    if model != str(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Only configured model is allowed")

    # Get request_id from middleware state
    request_id = getattr(request.state, "request_id", None) if request else None

    # Read file content first (needed for queue and streaming)
    content = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    # === INPUT GUARDRAILS ===
    # Run validation before queueing or streaming (need temp file for duration check)
    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(content)

        try:
            validate_input(
                filename=file.filename or "audio.wav",
                content=content,
                audio_path=temp_path,
            )
        except FileSizeError as e:
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="file_size_error",
            ).inc()
            raise HTTPException(status_code=413, detail=e.message)
        except FormatError as e:
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="format_error",
            ).inc()
            raise HTTPException(status_code=415, detail=e.message)
        except DurationError as e:
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="duration_error",
            ).inc()
            raise HTTPException(status_code=400, detail=e.message)
        except TimeoutError as e:
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="timeout_error",
            ).inc()
            raise HTTPException(status_code=408, detail=e.message)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    # === END INPUT GUARDRAILS ===

    if stream:
        stream_lock = state["stream_lock"]
        if stream_lock.locked():
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="streaming_busy",
            ).inc()
            raise HTTPException(
                status_code=429,
                detail={"error": "Only one streaming transcription is allowed at a time"},
            )

        await stream_lock.acquire()
        forced_language = _normalize_language(language)
        
        # Determine if we should use SSE format (vLLM-compatible)
        use_sse = stream and response_format == "json"

        async def stream_response():
            stream_temp_path = None
            stream_status = "success"
            start_time = time.perf_counter()
            accumulated_raw = ""
            accumulated_clean = ""
            emitted_final = False
            transcribe_id = f"transcribe-{uuid.uuid4()}"
            created_timestamp = int(time.time())

            try:
                with track_request("POST", "/v1/audio/transcriptions", request_id):
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        stream_temp_path = Path(tmp.name)
                        tmp.write(content)

                    with TimeoutContext(config.request_timeout_sec) as timeout_ctx:
                        token_stream = state["model"].generate(
                            str(stream_temp_path),
                            language=forced_language,
                            verbose=False,
                            stream=True,
                        )

                        for chunk in token_stream:
                            timeout_ctx.check_timeout()

                            if request is not None and await request.is_disconnected():
                                stream_status = "error"
                                break

                            # Get raw token text for SSE delta
                            raw_chunk_text = getattr(chunk, "text", "") or ""
                            accumulated_raw += raw_chunk_text
                            detected_language, parsed_text = _parse_asr_output(
                                accumulated_raw,
                                forced_language=forced_language,
                            )

                            delta_text = ""
                            if parsed_text.startswith(accumulated_clean):
                                delta_text = parsed_text[len(accumulated_clean) :]
                            elif parsed_text != accumulated_clean:
                                delta_text = parsed_text

                            accumulated_clean = parsed_text
                            is_final = bool(getattr(chunk, "is_final", False))
                            emitted_final = emitted_final or is_final

                            if use_sse:
                                # SSE format (vLLM-compatible)
                                if is_final:
                                    # Final chunk: empty content with finish_reason
                                    event_data = {
                                        "id": transcribe_id,
                                        "object": "transcription.chunk",
                                        "created": created_timestamp,
                                        "model": model,
                                        "choices": [
                                            {
                                                "delta": {
                                                    "content": "",  # Empty content for final
                                                },
                                                "finish_reason": "stop",
                                                "stop_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    break  # Stop streaming
                                else:
                                    # Regular chunk
                                    event_data = {
                                        "id": transcribe_id,
                                        "object": "transcription.chunk",
                                        "created": created_timestamp,
                                        "model": model,
                                        "choices": [
                                            {
                                                "delta": {
                                                    "content": raw_chunk_text,
                                                },
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                            else:
                                # Original ndjson format
                                yield (
                                    json.dumps(
                                        {
                                            "text": delta_text,
                                            "is_final": is_final,
                                            "language": detected_language,
                                            **(
                                                {"accumulated": accumulated_clean}
                                                if include_accumulated
                                                else {}
                                            ),
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )

                    if stream_status == "success" and not emitted_final:
                        detected_language, parsed_text = _parse_asr_output(
                            accumulated_raw,
                            forced_language=forced_language,
                        )
                        if use_sse:
                            # Emit final chunk with empty content, finish_reason="stop"
                            final_event = {
                                "id": transcribe_id,
                                "object": "transcription.chunk",
                                "created": created_timestamp,
                                "model": model,
                                "choices": [
                                    {
                                        "delta": {
                                            "content": "",
                                        },
                                        "finish_reason": "stop",
                                        "stop_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                        else:
                            yield (
                                json.dumps(
                                    {
                                        "text": "",
                                        "is_final": True,
                                        "language": detected_language,
                                        **(
                                            {"accumulated": parsed_text}
                                            if include_accumulated
                                            else {}
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
            except TimeoutError:
                stream_status = "error"
                raise
            except Exception:
                stream_status = "error"
                raise
            finally:
                REQUEST_DURATION.labels(
                    method="POST",
                    endpoint="/v1/audio/transcriptions",
                ).observe(time.perf_counter() - start_time)
                REQUEST_COUNT.labels(
                    method="POST",
                    endpoint="/v1/audio/transcriptions",
                    status=stream_status,
                ).inc()
                if stream_temp_path is not None and stream_temp_path.exists():
                    stream_temp_path.unlink()
                if stream_lock.locked():
                    stream_lock.release()

        media_type = "text/event-stream" if use_sse else "application/x-ndjson"
        return StreamingResponse(stream_response(), media_type=media_type)

    with track_request("POST", "/v1/audio/transcriptions", request_id):
        # === QUEUE ADMISSION ===
        # Try to add request to queue (non-blocking)
        queue = get_request_queue()
        success, queued_request = await queue.put(
            file_data=content,
            file_suffix=suffix,
            language=language,
            model=model,
            stream=stream,
            request_id=request_id,
        )

        if not success:
            # Queue is full - return 429 Too Many Requests
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="queue_full",
            ).inc()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Queue full - too many concurrent requests",
                    "max_queue_size": queue.max_size,
                    "retry_after": "30",
                },
            )

        if queued_request is None:
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="processing_error",
            ).inc()
            raise HTTPException(status_code=500, detail="Queue admission failed unexpectedly")

        # === WAIT FOR RESULT ===
        # Wait for the worker to process the request
        try:
            result = await queued_request.future
            # Record successful request
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="success",
            ).inc()
            return result
        except Exception as e:
            # Request processing failed
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/v1/audio/transcriptions",
                status="processing_error",
            ).inc()
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
