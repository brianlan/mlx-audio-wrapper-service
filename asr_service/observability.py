"""
Observability module for MLX Audio ASR Service.

Provides:
- JSON structured logging with request IDs
- Prometheus metrics endpoint at /metrics

Metrics:
- request_count: Total number of requests
- request_duration_seconds: Request duration histogram (p50/p95/p99)
- error_count: Total number of errors
- queue_depth: Current queue depth (requests in progress)
- gpu_memory_bytes: Current GPU memory usage
- active_requests: Number of currently active requests
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Generator, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

# Try to import mlx for GPU memory, but don't fail if unavailable
# These are runtime imports that will work when dependencies are installed
MLX_AVAILABLE = False
get_memory_pool_info = None  # type: ignore
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    # Check if function exists (not available in all MLX versions)
    if hasattr(mx, "get_memory_pool_info"):
        get_memory_pool_info = mx.get_memory_pool_info
except (ImportError, AttributeError):
    pass


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with request IDs."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request_id if present in extra
        request_id = getattr(record, "request_id", None)
        if request_id:
            log_data["request_id"] = request_id

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "request_id",
            ):
                if not key.startswith("_"):
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup JSON structured logging."""
    logger = logging.getLogger("mlx_asr")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logging()


# Prometheus Metrics
REQUEST_COUNT = Counter(
    "asr_request_count_total",
    "Total number of ASR requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "asr_request_duration_seconds",
    "ASR request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ERROR_COUNT = Counter(
    "asr_error_count_total",
    "Total number of errors",
    ["error_type"],
)

QUEUE_DEPTH = Gauge(
    "asr_queue_depth",
    "Current number of requests in progress",
)

GPU_MEMORY_BYTES = Gauge(
    "asr_gpu_memory_bytes",
    "Current GPU memory usage in bytes",
)

ACTIVE_REQUESTS = Gauge(
    "asr_active_requests",
    "Number of currently active requests",
)


def get_gpu_memory() -> Optional[int]:
    """Get current GPU memory usage in bytes."""
    if not MLX_AVAILABLE or get_memory_pool_info is None:
        return None
    try:
        # Get memory stats from MLX
        # mlx.core.get_memory_pool_info returns (allocated, reserved)
        memory_info = get_memory_pool_info()
        if memory_info:
            # Return reserved memory as that's what matters for allocation
            return int(memory_info[1]) if len(memory_info) > 1 else None
    except Exception:
        pass
    return None


def update_gpu_memory_metric() -> None:
    """Update GPU memory metric."""
    memory_bytes = get_gpu_memory()
    if memory_bytes is not None:
        GPU_MEMORY_BYTES.set(memory_bytes)


@contextmanager
def track_request(
    method: str,
    endpoint: str,
    request_id: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to track request metrics."""
    # Generate request ID if not provided
    req_id = request_id or str(uuid.uuid4())

    # Add request_id to logging context
    old_request_id = None
    if logger.handlers:
        for handler in logger.handlers:
            old_request_id = getattr(handler, "request_id", None)
            setattr(handler, "request_id", req_id)

    # Track metrics
    ACTIVE_REQUESTS.inc()
    QUEUE_DEPTH.inc()

    request_data = {
        "request_id": req_id,
        "method": method,
        "endpoint": endpoint,
    }

    try:
        yield request_data
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(
            f"Request failed: {str(e)}",
            extra={"request_id": req_id, "error_type": type(e).__name__},
            exc_info=True,
        )
        raise
    finally:
        ACTIVE_REQUESTS.dec()
        QUEUE_DEPTH.dec()

        # Remove request_id from logging context
        for handler in logger.handlers:
            if hasattr(handler, "request_id"):
                setattr(handler, "request_id", old_request_id)


def track_duration(method: str, endpoint: str):
    """Decorator to track request duration."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with REQUEST_DURATION.labels(method=method, endpoint=endpoint).time():
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def log_request(
    request: Request,
    status: str = "success",
    duration: Optional[float] = None,
) -> None:
    """Log request with structured data."""
    request_id = getattr(request.state, "request_id", None)

    log_data = {
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "client_host": request.client.host if request.client else None,
    }

    if request_id:
        log_data["request_id"] = request_id

    if duration:
        log_data["duration_seconds"] = duration

    # Filter out audio content and sensitive data
    # Note: Audio content should never be logged
    if status == "success":
        logger.info(f"Request completed: {request.method} {request.url.path}", extra=log_data)
    else:
        logger.warning(f"Request failed: {request.method} {request.url.path}", extra=log_data)


def get_metrics() -> bytes:
    """Generate Prometheus metrics in plaintext format."""
    # Update GPU memory before generating metrics
    update_gpu_memory_metric()
    return generate_latest()


# Request ID middleware for FastAPI
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to request state and logs."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Add to request headers for traceability
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response
