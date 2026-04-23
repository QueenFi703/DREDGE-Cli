"""
DREDGE Event Stream Handler
Provides a streaming batch processor for GitHub events so that multiple
events can be handled in a single CLI invocation rather than one per call.
"""
import asyncio
import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional


# Type alias for an event handler coroutine
EventHandler = Callable[[Dict[str, Any]], Any]


class EventQueue:
    """
    Thread-safe async queue for incoming GitHub events.

    Events are enqueued by the producer (e.g. the CLI) and drained by
    :class:`EventStreamProcessor`.
    """

    def __init__(self, maxsize: int = 0):
        """
        Args:
            maxsize: Maximum number of items held in the queue.  0 = unbounded.
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._enqueued: int = 0
        self._processed: int = 0

    async def put(self, event: Dict[str, Any]) -> None:
        """Enqueue an event."""
        await self._queue.put(event)
        self._enqueued += 1

    def put_nowait(self, event: Dict[str, Any]) -> None:
        """Enqueue without waiting (raises ``asyncio.QueueFull`` if full)."""
        self._queue.put_nowait(event)
        self._enqueued += 1

    async def get(self) -> Dict[str, Any]:
        """Dequeue the next event, waiting if necessary."""
        item = await self._queue.get()
        self._processed += 1
        return item

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all enqueued items have been processed."""
        await self._queue.join()

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "enqueued": self._enqueued,
            "processed": self._processed,
            "pending": self._queue.qsize(),
        }


class EventStreamProcessor:
    """
    High-throughput streaming processor for GitHub events.

    Events are consumed from an :class:`EventQueue` by a configurable
    number of async worker coroutines so that multiple events are handled
    concurrently within a single OS process.
    """

    def __init__(
        self,
        handler: EventHandler,
        num_workers: int = 4,
        batch_size: int = 16,
        drain_timeout: float = 5.0,
    ):
        """
        Args:
            handler: Async (or sync) callable invoked for each event.
            num_workers: Number of concurrent worker coroutines.
            batch_size: Number of events each worker attempts to process
                        before yielding.
            drain_timeout: Seconds to wait for the queue to drain before
                           stopping workers.
        """
        self.handler = handler
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.drain_timeout = drain_timeout
        self.queue: EventQueue = EventQueue()

        self._results: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, event: Dict[str, Any]) -> None:
        """Add an event to the processing queue (non-blocking)."""
        self.queue.put_nowait(event)

    def enqueue_many(self, events: List[Dict[str, Any]]) -> None:
        """Add multiple events to the queue."""
        for event in events:
            self.enqueue(event)

    async def run_async(self) -> Dict[str, Any]:
        """
        Start workers, drain the queue, and return a summary.

        Returns:
            Summary dictionary with throughput and error statistics.
        """
        self._start_time = time.perf_counter()
        workers = [
            asyncio.create_task(self._worker(worker_id=i))
            for i in range(self.num_workers)
        ]
        # Wait for the queue to be fully processed
        try:
            await asyncio.wait_for(self.queue.join(), timeout=self.drain_timeout)
        except asyncio.TimeoutError:
            pass  # Partial drain – workers will stop on the sentinel below

        # Signal workers to stop
        for _ in workers:
            self.queue.put_nowait(_SENTINEL)  # type: ignore[arg-type]

        await asyncio.gather(*workers, return_exceptions=True)
        self._end_time = time.perf_counter()

        return self._build_summary()

    def run(self) -> Dict[str, Any]:
        """Synchronous convenience wrapper around :meth:`run_async`."""
        return asyncio.run(self.run_async())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        """Consumer coroutine that drains the queue."""
        while True:
            event = await self.queue.get()
            if event is _SENTINEL:
                self.queue.task_done()
                break
            try:
                if asyncio.iscoroutinefunction(self.handler):
                    result = await self.handler(event)
                else:
                    result = self.handler(event)
                self._results.append(
                    {"worker": worker_id, "event": event, "result": result}
                )
            except Exception as exc:  # noqa: BLE001
                self._errors.append(
                    {
                        "worker": worker_id,
                        "event": event,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
            finally:
                self.queue.task_done()

    def _build_summary(self) -> Dict[str, Any]:
        elapsed = (
            (self._end_time or 0) - (self._start_time or 0)
        )
        total = len(self._results) + len(self._errors)
        return {
            "total_processed": total,
            "successful": len(self._results),
            "errors": len(self._errors),
            "elapsed_seconds": elapsed,
            "throughput_per_sec": total / elapsed if elapsed > 0 else 0,
            "queue_stats": self.queue.stats,
        }


# Sentinel object used to signal workers to exit
_SENTINEL: Any = object()


# ---------------------------------------------------------------------------
# CLI helper – stream events from stdin or a JSON file
# ---------------------------------------------------------------------------


def process_events_from_stdin(
    handler: EventHandler,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Read newline-delimited JSON events from stdin and process them.

    Each line must be a complete JSON object representing one GitHub event.
    Empty lines are skipped.

    Args:
        handler: Callable invoked per event (sync or async).
        num_workers: Number of concurrent workers.

    Returns:
        Processing summary.
    """
    events: List[Dict[str, Any]] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip malformed lines

    processor = EventStreamProcessor(handler=handler, num_workers=num_workers)
    processor.enqueue_many(events)
    return processor.run()


def process_events_from_list(
    events: List[Dict[str, Any]],
    handler: EventHandler,
    num_workers: int = 4,
    drain_timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Process a pre-collected list of events.

    Args:
        events: List of event dictionaries.
        handler: Callable invoked per event.
        num_workers: Number of concurrent workers.
        drain_timeout: Seconds to wait for queue drain.

    Returns:
        Processing summary.
    """
    processor = EventStreamProcessor(
        handler=handler,
        num_workers=num_workers,
        drain_timeout=drain_timeout,
    )
    processor.enqueue_many(events)
    return processor.run()
