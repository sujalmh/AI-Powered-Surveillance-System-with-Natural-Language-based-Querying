"""
Run blocking (sync) code in the default thread pool so the event loop is not blocked.
Use in async route handlers for Mongo, file I/O, or CPU-heavy work.
"""
import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_sync(sync_fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable in the thread pool. Await from async route handlers."""
    return await asyncio.to_thread(sync_fn, *args, **kwargs)
