"""
Bounded task queue for heavy background jobs (e.g. auto-indexing).
Uses ThreadPoolExecutor for backpressure and error logging instead of unbounded threads.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_log = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai-tasks")


def _done_callback(fut: Any) -> None:
    try:
        exc = fut.exception()
        if exc is not None:
            _log.error("Task error: %s", exc, exc_info=True)
    except Exception:
        pass


def submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Submit a task to the bounded pool. Returns a Future."""
    future = _pool.submit(fn, *args, **kwargs)
    future.add_done_callback(_done_callback)
    return future


def shutdown(wait: bool = False, cancel_futures: bool = True) -> None:
    """Shut down the pool. Cancel pending futures and don't wait for running tasks."""
    try:
        _pool.shutdown(wait=wait, cancel_futures=cancel_futures)
    except TypeError:
        # Python < 3.9 doesn't support cancel_futures
        _pool.shutdown(wait=wait)
    except Exception:
        pass
