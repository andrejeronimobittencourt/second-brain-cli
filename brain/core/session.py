"""Vault session file paths and history load/save."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from brain.core.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)

_SAFE_SESSION_NAME = re.compile(r'[^a-zA-Z0-9._-]+')


def sanitize_session_name(name: str) -> str:
    """Return a filesystem-safe session slug."""
    cleaned = _SAFE_SESSION_NAME.sub('_', name.strip())
    return cleaned[:64] or 'session'


def resolve_history_path(
    vault_path: Path,
    history_filename: str,
    session_name: str = '',
) -> Path:
    """
    Resolve where conversation history is stored inside the vault.

    Default: ``{vault}/{history_filename}``.
    Named session: ``{vault}/.agent_sessions/{name}.json``.
    """
    if session_name.strip():
        slug = sanitize_session_name(session_name)
        return vault_path / '.agent_sessions' / f'{slug}.json'
    filename = history_filename.strip() or '.agent_history.json'
    return vault_path / filename


def _try_load_json(path: Path) -> list[dict[str, Any]] | None:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, list):
        return None
    return [m for m in raw if isinstance(m, dict)]


def load_history(
    path: Path,
    *,
    max_messages: int,
    truncate_fn: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]],
    on_error: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Load non-system messages from a JSON array session file."""
    def _report(msg: str) -> None:
        if on_error is not None:
            on_error(msg)

    if not path.is_file():
        tmp = path.with_suffix(path.suffix + '.tmp')
        if tmp.is_file():
            recovered = _try_load_json(tmp)
            if recovered is not None:
                logger.info('Recovered session from temp file %s', tmp)
                return truncate_fn(recovered, max_messages)
        return []
    loaded = _try_load_json(path)
    if loaded is None:
        tmp = path.with_suffix(path.suffix + '.tmp')
        recovered = _try_load_json(tmp) if tmp.is_file() else None
        if recovered is not None:
            _report(
                f"Session file '{path}' is corrupted; recovered from temp copy.",
            )
            return truncate_fn(recovered, max_messages)
        _report(
            f"Session file '{path}' is corrupted (invalid JSON) and will be ignored.",
        )
        return []
    return truncate_fn(loaded, max_messages)


def save_history(
    path: Path,
    messages: list[dict[str, Any]],
    *,
    max_messages: int,
    truncate_fn: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]],
    on_error: Callable[[str], None] | None = None,
) -> None:
    """Persist non-system messages as a JSON array (atomic write)."""
    non_system = [m for m in messages if m.get('role') != 'system']
    non_system = truncate_fn(non_system, max_messages)
    payload = json.dumps(non_system, ensure_ascii=False, indent=2)
    try:
        atomic_write_text(path, payload)
    except OSError as exc:
        if on_error is not None:
            on_error(f'Session could not be saved: {exc}')
