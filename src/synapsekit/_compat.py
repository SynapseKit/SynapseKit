from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ._loop import install_fast_loop

if TYPE_CHECKING:
    from collections.abc import Coroutine

T = TypeVar("T")

install_fast_loop()


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine synchronously.
    Works both inside and outside a running event loop.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Running inside an existing loop (e.g., Jupyter).
        # Use a new thread with its own loop to avoid deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
