"""async_pipeline.py — Asynchronous background compression runner.

After each LLM turn, call ``on_turn_complete(turn_idx)`` to (optionally)
schedule a background compression pass.  The pass runs on a dedicated worker
thread so it never blocks the main LLM inference loop.

Before the *next* LLM call, call ``wait_if_needed()`` to ensure any in-flight
compression job has finished — so the context returned by
``ContextStore.get_context_for_llm()`` reflects the latest compressed state.

Design notes
------------
- A single ``ThreadPoolExecutor`` with ``max_workers=1`` serialises all
  compression jobs (FIFO).  This avoids race conditions in ``ContextStore``
  and mirrors the single-worker-process design in the proposal.
- The executor is created lazily on first use and kept alive for the lifetime
  of the ``AsyncPipeline`` instance.
- ``shutdown()`` gracefully drains in-flight jobs and shuts down the pool.

Usage::

    pipeline = AsyncPipeline(context_store=store, compressor=compressor)
    pipeline.on_turn_complete(turn_idx=5)   # schedules if budget exceeded
    ...
    pipeline.wait_if_needed()               # called just before next LLM call
    context = store.get_context_for_llm()
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_store import ContextStore
    from .compressor import HeuristicCompressor

logger = logging.getLogger(__name__)


class AsyncPipeline:
    """
    Manages background compression jobs, decoupled from the main LLM loop.

    Args:
        context_store: The shared ``ContextStore`` instance.
        compressor:    The ``HeuristicCompressor`` instance.
        compress_always: If True, schedule compression after every turn
                         regardless of whether the budget is exceeded.
                         Useful for demos and testing.
    """

    def __init__(
        self,
        context_store: ContextStore,
        compressor: HeuristicCompressor,
        compress_always: bool = False,
    ) -> None:
        self._store = context_store
        self._compressor = compressor
        self._compress_always = compress_always

        self._executor: ThreadPoolExecutor | None = None
        self._current_future: Future[dict] | None = None
        self._lock = threading.Lock()

        # Metrics
        self.jobs_scheduled: int = 0
        self.jobs_completed: int = 0
        self.total_compress_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_turn_complete(self, turn_idx: int) -> bool:
        """
        Called after each LLM turn completes.

        If the context budget is exceeded (or ``compress_always`` is True),
        schedules a background compression pass for the oldest uncompressed
        non-frozen segment.

        Args:
            turn_idx: The index of the turn that just completed.

        Returns:
            True if a compression job was scheduled; False otherwise.
        """
        should_compress = self._compress_always or self._store.budget_exceeded()
        if not should_compress:
            logger.debug("turn %d — budget OK, skipping compression", turn_idx)
            return False

        candidate = self._store.get_oldest_uncompressed_segment()
        if candidate is None:
            logger.debug("turn %d — no compressible segments available", turn_idx)
            return False

        segment, is_recent = candidate
        logger.debug(
            "turn %d — scheduling compression for segment %d (recent=%s)",
            turn_idx, segment.segment_id, is_recent,
        )
        self._schedule(segment.segment_id, is_recent)
        return True

    def wait_if_needed(self) -> None:
        """
        Block until any in-flight compression job finishes.

        Call this immediately before each LLM call to ensure the context is
        fully up-to-date.  If no job is running, returns immediately.
        """
        with self._lock:
            future = self._current_future

        if future is not None and not future.done():
            logger.debug("waiting for in-flight compression job…")
            t0 = time.monotonic()
            future.result()  # blocks
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.debug("waited %.1f ms for compression", elapsed_ms)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the background executor.

        Args:
            wait: If True, block until all pending jobs finish before returning.
        """
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
                self._executor = None

    def stats(self) -> dict[str, float | int]:
        """Return a snapshot of pipeline metrics."""
        return {
            "jobs_scheduled": self.jobs_scheduled,
            "jobs_completed": self.jobs_completed,
            "avg_compress_ms": (
                self.total_compress_ms / self.jobs_completed
                if self.jobs_completed > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create and return the worker thread pool."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="compressor",
                )
            return self._executor

    def _schedule(self, segment_id: int, is_recent: bool) -> None:
        """Submit a compression job to the background worker."""
        executor = self._get_executor()
        with self._lock:
            # Wait for any previous job before submitting a new one
            # (serialises jobs — mirrors the FIFO proposal design)
            if self._current_future is not None and not self._current_future.done():
                # Don't pile up jobs; the caller will check on next turn
                logger.debug("previous job still running, deferring segment %d", segment_id)
                return

            future = executor.submit(self._compression_job, segment_id, is_recent)
            self._current_future = future
            self.jobs_scheduled += 1

    def _compression_job(self, segment_id: int, is_recent: bool) -> dict:
        """
        The actual compression work that runs on the background thread.

        Looks up the segment, runs the compressor, and writes the result back
        to the ContextStore.

        Returns:
            A dict with job metadata and the CompressionResult summary.
        """
        t0 = time.monotonic()
        result_meta: dict = {
            "segment_id": segment_id,
            "success": False,
            "elapsed_ms": 0.0,
        }

        try:
            # Re-fetch the segment under the store's lock
            candidate = None
            for seg, recent_flag in self._store.get_compressible_segments():
                if seg.segment_id == segment_id:
                    candidate = (seg, recent_flag)
                    break

            if candidate is None:
                logger.warning("segment %d disappeared before compression", segment_id)
                return result_meta

            segment, actual_is_recent = candidate

            # Determine target ratio from the segment's current depth and age
            from .context_store import RECENT_MAX_RATIO, OLD_MAX_RATIO
            max_ratio = RECENT_MAX_RATIO if actual_is_recent else OLD_MAX_RATIO
            # Start at max_ratio / (depth + 1) so each re-compression is gentler
            target_ratio = max_ratio / (segment.compression_depth + 1)
            target_ratio = max(1.5, target_ratio)  # never compress less than 1.5x

            raw_text = segment.effective_text
            result = self._compressor.compress(
                text=raw_text,
                target_ratio=target_ratio,
                is_recent=actual_is_recent,
                max_ratio=max_ratio,
            )

            success = self._store.apply_compression(
                segment_id=segment_id,
                compressed_text=result.compressed_text,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000
            self.jobs_completed += 1
            self.total_compress_ms += elapsed_ms

            result_meta.update({
                "success": success,
                "elapsed_ms": round(elapsed_ms, 1),
                "method": result.method,
                "actual_ratio": round(result.actual_ratio, 2),
                "fidelity_overlap": round(result.fidelity_overlap, 3),
            })

            logger.info(
                "compressed segment %d: %s (%.1f ms)",
                segment_id, result.summary_line(), elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            result_meta["elapsed_ms"] = round(elapsed_ms, 1)
            result_meta["error"] = str(exc)
            logger.exception("error compressing segment %d", segment_id)

        return result_meta
