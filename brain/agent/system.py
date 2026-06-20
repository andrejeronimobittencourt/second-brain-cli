"""System prompt assembly for the agent turn pipeline."""

from __future__ import annotations

import re
from typing import Any

from brain.core.context import get_context
from brain.core.system_prompt import build_system_prompt

_FOLLOWUP_INVITE_MARKERS: tuple[str, ...] = (
    'let me know',
    'would you like',
    'which folder',
    'which topic',
    'explore next',
    'want me to',
    'tell me if',
    'feel free to',
    "if you'd like",
    'if you would like',
    'shall i',
    'do you want',
    'anything else',
)


def effective_system_prompt() -> str:
    """Build the system message for Ollama (layered prompt + runtime footer)."""
    ctx = get_context()
    base = build_system_prompt(user=ctx.user, vault_path=ctx.vault_path)
    if not ctx.print_mode:
        return base
    model = ctx.defaults.model
    parts: list[str] = []
    preamble = model.print_mode_system_preamble.strip()
    if preamble:
        parts.append(preamble)
    parts.append(base)
    suffix = model.print_mode_one_shot_instructions.strip()
    if suffix:
        parts.append(suffix)
    return '\n\n'.join(parts)


def wrap_print_mode_user_message(user_input: str) -> str:
    """Add a one-shot constraint to the user message for ``--print``."""
    text = user_input.strip()
    if not text:
        return text
    return (
        f'{text}\n\n'
        '[One-shot CLI: reply with the full answer only. '
        'Do not ask follow-up questions or invite further exploration.]'
    )


def _looks_like_followup_invite(paragraph: str) -> bool:
    lower = paragraph.lower()
    if not lower.endswith('?'):
        return False
    return any(marker in lower for marker in _FOLLOWUP_INVITE_MARKERS)


def sanitize_print_mode_answer(text: str) -> str:
    """
    Drop a trailing conversational invite from one-shot CLI output.

    Safety net when the model ignores print-mode instructions.
    """
    stripped = text.strip()
    if not stripped:
        return stripped
    paragraphs = re.split(r'\n\s*\n', stripped)
    while len(paragraphs) >= 2:
        last = paragraphs[-1].strip()
        if not _looks_like_followup_invite(last):
            break
        paragraphs.pop()
    return '\n\n'.join(paragraphs).strip()


def empty_answer_fallback() -> str:
    """Plain fallback when the model produces no assistant text."""
    ctx = get_context()
    model = ctx.defaults.model
    if ctx.print_mode:
        return model.empty_answer_print_message
    return model.empty_answer_repl_message


def refresh_system_message(messages: list[dict[str, Any]]) -> None:
    """Update the system message with a fresh runtime footer."""
    if messages and messages[0].get('role') == 'system':
        messages[0]['content'] = effective_system_prompt()
    else:
        messages.insert(0, {'role': 'system', 'content': effective_system_prompt()})
