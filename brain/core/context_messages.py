"""Chat message list trimming for context windows and session persistence."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def split_messages_pair_aware(
    messages: list[dict[str, Any]],
    cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Partition *messages* into ``(dropped_prefix, kept_suffix)``.

    The suffix keeps at most *cap* messages; if the slice begins with a stray
    ``tool`` message (no preceding assistant in the window), leading tools are
    removed from the suffix.
    """
    if len(messages) <= cap:
        return [], messages
    trimmed = messages[-cap:]
    while trimmed and trimmed[0].get('role') == 'tool':
        trimmed.pop(0)
    n_kept = len(trimmed)
    dropped = messages[: len(messages) - n_kept]
    if dropped:
        logger.debug(
            'Context trimmed: dropped %d oldest messages (cap=%d)',
            len(dropped),
            cap,
        )
    return dropped, trimmed


def truncate_messages_pair_aware(
    messages: list[dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    """Keep the last *cap* messages, dropping orphaned leading tool results."""
    _, kept = split_messages_pair_aware(messages, cap)
    return kept
