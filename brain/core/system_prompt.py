"""Layered system prompt assembly for Ollama."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from brain.core.defaults import default_system_prompt
from brain.core.system_context import system_context_footer
from brain.core.user_config import UserConfig


def build_system_prompt(
    *,
    user: UserConfig,
    vault_path: Path,
    now: datetime | None = None,
) -> str:
    """
    Build the effective system message.

    Order: base (custom or default) → ``vault_instructions`` → runtime footer.
    """
    custom = (user.system_prompt or '').strip()
    add = (user.vault_instructions or '').strip()
    base = custom if custom else default_system_prompt()
    parts = [base]
    if add:
        parts.append(add)
    parts.append(system_context_footer(vault_path=vault_path, now=now))
    return '\n\n'.join(parts)
