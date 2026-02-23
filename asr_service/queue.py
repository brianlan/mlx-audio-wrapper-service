"""Request queue module for MLX Audio ASR Service.

Provides a bounded asyncio queue with overflow handling for backpressure.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from asr_service.config import config
from asr_service.observability import logger


@dataclass
class QueuedRequest:
    """Represents a request in the queue."""

    request_id: str
    file_data: bytes
    file_suffix: str
    language: Optional[str]
    model: str
    stream: bool
    future: asyncio.Future


class RequestQueue:
    """Bounded request queue with overflow handling.

    This queue:
    - Has a maximum size driven by config (queue_max_size)
    - Uses non-blocking put_nowait() for admission
    - Returns False (rejects) when full
    - Note: Queue depth is tracked by observability.track_request (in-flight requests)
    """

    def __init__(self, max_size: Optional[int] = None):
        """Initialize the request queue.

        Args:
            max_size: Maximum queue size. Defaults to config.queue_max_size.
        """
        self._max_size = max_size or config.queue_max_size
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=self._max_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._process_handler: Optional[Callable[[QueuedRequest], Awaitable[Any]]] = None
        self._running = False

    @property
    def max_size(self) -> int:
        """Return the maximum queue size."""
        return self._max_size

    @property
    def current_size(self) -> int:
        """Return the current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if the queue is at capacity."""
        return self._queue.full()

    async def put(
        self,
        file_data: bytes,
        file_suffix: str,
        language: Optional[str],
        model: str,
        stream: bool,
        request_id: Optional[str] = None,
    ) -> tuple[bool, Optional[QueuedRequest]]:
        """Try to add a request to the queue (non-blocking).

        Args:
            file_data: Audio file bytes
            file_suffix: File extension (e.g., '.wav')
            language: Language hint
            model: Model path
            stream: Whether to stream the response
            request_id: Optional request ID (generated if not provided)

        Returns:
            Tuple of (success, queued_request). If success is False,
            queued_request is None and caller should return 429.
        """
        req_id = request_id or str(uuid.uuid4())
        queued_request = QueuedRequest(
            request_id=req_id,
            file_data=file_data,
            file_suffix=file_suffix,
            language=language,
            model=model,
            stream=stream,
            future=asyncio.Future(),
        )

        try:
            # Non-blocking put - raises QueueFull if at capacity
            self._queue.put_nowait(queued_request)
            logger.info(
                f"Request {req_id} added to queue",
                extra={"request_id": req_id, "queue_size": self._queue.qsize()},
            )
            return True, queued_request
        except asyncio.QueueFull:
            logger.warning(
                f"Queue full - rejecting request {req_id}",
                extra={
                    "request_id": req_id,
                    "queue_size": self._queue.qsize(),
                    "max_size": self._max_size,
                },
            )
            return False, None

    async def get(self) -> QueuedRequest:
        """Get a request from the queue (blocking).

        This method blocks until a request is available.

        Returns:
            The next queued request.
        """
        request = await self._queue.get()
        return request

    def task_done(self) -> None:
        """Mark a task as done (called after processing)."""
        self._queue.task_done()

    async def start_worker(self, process_handler: Callable[[QueuedRequest], Awaitable[Any]]) -> None:
        """Start the background worker that processes queued requests.

        Args:
            process_handler: Async function to process each queued request.
        """
        self._process_handler = process_handler
        self._running = True

        async def worker():
            while self._running:
                try:
                    request = await self.get()
                    try:
                        result = await self._process_handler(request)
                        if not request.future.done():
                            request.future.set_result(result)
                    except Exception as e:
                        if not request.future.done():
                            request.future.set_exception(e)
                    finally:
                        self.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker error: {e}", exc_info=True)

        self._worker_task = asyncio.create_task(worker())
        logger.info("Request queue worker started")

    async def stop_worker(self, timeout: float = 30.0) -> None:
        """Stop the background worker gracefully.

        Args:
            timeout: Maximum time to wait for worker to finish.
        """
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(f"Worker did not stop within {timeout}s")

        logger.info("Request queue worker stopped")

    async def drain(self, timeout: float = 30.0) -> int:
        """Drain the queue (for graceful shutdown).

        Args:
            timeout: Maximum time to wait for queue to drain.

        Returns:
            Number of requests remaining in queue.
        """
        if self._queue.empty():
            return 0

        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        remaining = self._queue.qsize()
        return remaining


# Global request queue instance
request_queue: Optional[RequestQueue] = None


def get_request_queue() -> RequestQueue:
    """Get the global request queue instance.

    Returns:
        The global RequestQueue instance.
    """
    global request_queue
    if request_queue is None:
        request_queue = RequestQueue()
    return request_queue
