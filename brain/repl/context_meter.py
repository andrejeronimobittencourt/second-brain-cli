"""Shared context-fill meter for ``/stats``."""

from __future__ import annotations

from typing import Any


def non_system_message_count(messages: list[dict[str, Any]]) -> int:
    """Count conversation messages excluding the system prompt."""
    return sum(1 for m in messages if m.get('role') != 'system')


def resolve_context_meter(
    *,
    last_prompt_tokens: int,
    num_ctx: int,
    message_count: int,
    message_cap: int,
) -> tuple[int, int]:
    """
    Return ``(used, total)`` for context usage in ``/stats``.

    Uses Ollama prompt tokens when ``num_ctx`` is known, otherwise non-system
    message count versus ``max_context_messages``.
    """
    if num_ctx > 0:
        return last_prompt_tokens, num_ctx
    return message_count, message_cap
