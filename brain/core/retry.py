"""Retry wrapper for transient local Ollama connection failures."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Callable
from typing import TypeVar

from brain.core.defaults import RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar('T')


def is_retryable_connection_error(exc: BaseException) -> bool:
    """
    Return True for local blips worth retrying.

    Does not retry context overflow or application logic errors.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, 'errno', None) in {
        10061,  # WSAECONNREFUSED on Windows
        111,    # ECONNREFUSED on Linux
    }:
        return True
    msg = str(exc).lower()
    needles = (
        'connection refused',
        'connection reset',
        'connection aborted',
        'failed to connect',
        'timed out',
        'timeout',
        '502',
        '503',
        'temporarily unavailable',
    )
    return any(n in msg for n in needles)


def retry_call(fn: Callable[[], T], policy: RetryPolicy) -> T:
    """
    Call *fn* up to ``policy.max_attempts`` times with exponential backoff.

    Re-raises the last exception when all attempts fail.
    """
    if not policy.enabled or policy.max_attempts < 2:
        return fn()

    delay_ms = policy.base_delay_ms
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if not is_retryable_connection_error(exc):
                raise
            if attempt >= policy.max_attempts:
                break
            logger.warning(
                'Ollama call failed (attempt %d/%d): %s — retrying in %dms',
                attempt,
                policy.max_attempts,
                exc,
                delay_ms,
            )
            time.sleep(delay_ms / 1000.0)
            delay_ms = min(delay_ms * 2, policy.max_delay_ms)

    assert last_exc is not None
    raise last_exc


def call_with_timeout(fn: Callable[[], T], seconds: float) -> T:
    """
    Run blocking *fn* on a worker thread with a wall-clock timeout.

    Raises ``TimeoutError`` when *seconds* elapse before *fn* completes.
    """
    if seconds <= 0:
        return fn()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f'Operation timed out after {seconds:g}s',
            ) from exc
