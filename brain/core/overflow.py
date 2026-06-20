"""Ollama-only context overflow detection."""

from __future__ import annotations

import re
from typing import Final

# Patterns seen from Ollama / llama.cpp deployments (Ollama-only error strings).
_OLLAMA_OVERFLOW_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r'prompt too long', re.IGNORECASE),
    re.compile(r'exceeds the available context size', re.IGNORECASE),
    re.compile(r'context length exceeded', re.IGNORECASE),
    re.compile(r'too many tokens', re.IGNORECASE),
    re.compile(r'token limit exceeded', re.IGNORECASE),
    re.compile(r'context[_ ]length[_ ]exceeded', re.IGNORECASE),
)


def is_context_overflow_message(text: str) -> bool:
    """Return True when *text* looks like an Ollama context-window error."""
    if not text:
        return False
    return any(p.search(text) for p in _OLLAMA_OVERFLOW_PATTERNS)


def is_token_budget_exceeded(
    prompt_tokens: int,
    num_ctx: int,
    *,
    fill_ratio: float = 1.0,
) -> bool:
    """Return True when prompt tokens meet or exceed the model budget."""
    if num_ctx <= 0 or prompt_tokens <= 0:
        return False
    if prompt_tokens >= num_ctx:
        return True
    return (prompt_tokens / num_ctx) >= fill_ratio


def is_silent_overflow(
    prompt_tokens: int,
    num_ctx: int,
    assistant_content: str,
    *,
    fill_ratio: float = 0.95,
) -> bool:
    """
    Some Ollama stacks fill the context without a clear error string.

    Treat a near-full prompt with an empty assistant reply as overflow.
    """
    if (assistant_content or '').strip():
        return False
    return is_token_budget_exceeded(prompt_tokens, num_ctx, fill_ratio=fill_ratio)


def is_context_overflow(
    *,
    error: BaseException | str | None = None,
    message_text: str = '',
    prompt_tokens: int = 0,
    num_ctx: int = 0,
    assistant_content: str = '',
    silent_fill_ratio: float = 0.95,
) -> bool:
    """
    Decide whether the last LLM interaction hit a context limit.

    Checks, in order: exception text, assistant error text, token fill,
    silent overflow heuristic.
    """
    if error is not None:
        if is_context_overflow_message(str(error)):
            return True
    if is_context_overflow_message(message_text):
        return True
    if is_token_budget_exceeded(prompt_tokens, num_ctx, fill_ratio=1.0):
        return True
    return is_silent_overflow(
        prompt_tokens,
        num_ctx,
        assistant_content,
        fill_ratio=silent_fill_ratio,
    )
