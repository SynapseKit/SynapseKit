"""TimeBasedAutoResume — auto-resume interrupted graphs after a time delay."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpointers.base import BaseCheckpointer
    from .compiled import CompiledGraph


class TimeBasedAutoResume:
    """Auto-resume interrupted graphs after a configurable time delay.

    Usage::

        from synapsekit.graph import TimeBasedAutoResume, InMemoryCheckpointer

        checkpointer = InMemoryCheckpointer()
        auto_resume = TimeBasedAutoResume(
            compiled_graph=compiled,
            checkpointer=checkpointer,
            resume_after_seconds=3600,
            max_retries=3,
            poll_interval=60.0,
        )

        result = await auto_resume.start("thread-1", {"input": "hello"})

    Args:
        compiled_graph: A compiled graph with a ``resume(graph_id, checkpointer)`` method.
        checkpointer: A checkpointer with ``.load(thread_id)`` / ``.save(thread_id, step, state)``.
        resume_after_seconds: Seconds to wait before retrying after an interruption (default: 3600).
        max_retries: Maximum number of auto-resume attempts per thread (default: 3).
        poll_interval: Background poll interval in seconds (default: 60.0).
    """

    def __init__(
        self,
        compiled_graph: CompiledGraph,
        checkpointer: BaseCheckpointer,
        resume_after_seconds: float = 3600.0,
        max_retries: int = 3,
        poll_interval: float = 60.0,
    ) -> None:
        self.compiled_graph = compiled_graph
        self.checkpointer = checkpointer
        self.resume_after_seconds = resume_after_seconds
        self.max_retries = max_retries
        self.poll_interval = poll_interval

        # thread_id -> {state, retries, interrupted_at, cancelled, result, error}
        self._threads: dict[str, dict[str, Any]] = {}
        self._poll_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self, thread_id: str, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Run the graph; auto-schedule resume on failure.

        Returns the final state dict on success.
        Raises the last exception if all retries are exhausted.
        """
        self._ensure_poller()

        self._threads[thread_id] = {
            "state": dict(initial_state),
            "retries": 0,
            "interrupted_at": None,
            "cancelled": False,
            "result": None,
            "error": None,
        }

        try:
            result = await self.compiled_graph.run(
                initial_state, checkpointer=self.checkpointer, graph_id=thread_id
            )
            self._threads[thread_id]["result"] = result
            self._threads[thread_id]["interrupted_at"] = None
            return result
        except Exception as exc:
            self._threads[thread_id]["interrupted_at"] = time.time()
            self._threads[thread_id]["error"] = exc
            self._threads[thread_id]["state"] = dict(initial_state)
            raise

    async def cancel(self, thread_id: str) -> None:
        """Cancel a pending auto-resume for *thread_id*."""
        if thread_id in self._threads:
            self._threads[thread_id]["cancelled"] = True

    async def status(self, thread_id: str) -> dict[str, Any]:
        """Return status for a thread.

        Returns::

            {
                "thread_id": str,
                "state": dict | None,
                "retries": int,
                "next_resume_at": float | None,  # Unix timestamp or None
                "cancelled": bool,
                "completed": bool,
            }
        """
        info = self._threads.get(thread_id)
        if info is None:
            return {
                "thread_id": thread_id,
                "state": None,
                "retries": 0,
                "next_resume_at": None,
                "cancelled": False,
                "completed": False,
            }

        next_resume_at: float | None = None
        if info["interrupted_at"] is not None and not info["cancelled"]:
            next_resume_at = info["interrupted_at"] + self.resume_after_seconds

        return {
            "thread_id": thread_id,
            "state": info.get("state"),
            "retries": info["retries"],
            "next_resume_at": next_resume_at,
            "cancelled": info["cancelled"],
            "completed": info.get("result") is not None,
        }

    # ------------------------------------------------------------------ #
    # Background poller
    # ------------------------------------------------------------------ #

    def _ensure_poller(self) -> None:
        """Start background polling task if not already running."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.ensure_future(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Background task: check threads that need auto-resume."""
        while True:
            await asyncio.sleep(self.poll_interval)
            now = time.time()
            for thread_id, info in list(self._threads.items()):
                if info["cancelled"]:
                    continue
                if info["result"] is not None:
                    continue
                interrupted_at = info["interrupted_at"]
                if interrupted_at is None:
                    continue
                if now - interrupted_at < self.resume_after_seconds:
                    continue
                if info["retries"] >= self.max_retries:
                    continue
                # Time to attempt resume
                info["retries"] += 1
                info["interrupted_at"] = None
                try:
                    result = await self.compiled_graph.resume(thread_id, self.checkpointer)
                    info["result"] = result
                    info["error"] = None
                except Exception as exc:
                    info["interrupted_at"] = time.time()
                    info["error"] = exc


def schedule_resume(
    self: CompiledGraph,
    thread_id: str,
    checkpointer: BaseCheckpointer,
    after_seconds: float = 3600.0,
) -> TimeBasedAutoResume:
    """Convenience method: create a TimeBasedAutoResume and schedule a resume.

    This is monkey-patched onto CompiledGraph as ``schedule_resume``.

    Returns the ``TimeBasedAutoResume`` instance (background task already started).
    """
    auto = TimeBasedAutoResume(
        compiled_graph=self,
        checkpointer=checkpointer,
        resume_after_seconds=after_seconds,
    )
    auto._ensure_poller()
    return auto
