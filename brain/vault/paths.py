"""Symlink-safe vault path resolution using realpath jail."""

from __future__ import annotations

import os
from pathlib import Path

from brain.core.context import get_context

_cached_root: Path | None = None
_cached_vault_config: Path | None = None


def is_hidden_vault_path(path: Path) -> bool:
    """Return True when any path component is dot-prefixed (e.g. ``.agent``)."""
    return any(part.startswith('.') for part in path.parts)


def _ensure_cache() -> None:
    global _cached_root, _cached_vault_config
    cfg = get_context().vault_path
    if _cached_vault_config != cfg:
        _cached_root = None
        _cached_vault_config = cfg
    if _cached_root is None:
        _cached_root = Path(os.path.realpath(cfg))


def vault_root() -> Path:
    """Return the vault root as a canonical real path (cached per process)."""
    _ensure_cache()
    assert _cached_root is not None
    return _cached_root


def reject_parent_segments(relative: str) -> bool:
    """Return True when *relative* contains a ``..`` path segment."""
    normalized = relative.strip().replace('\\', '/').strip('/')
    if not normalized:
        return False
    return any(part == '..' for part in normalized.split('/'))


def assert_in_vault(path: Path) -> Path:
    """
    Ensure *path* resolves inside the vault root.

    Returns the real path. Raises ``ValueError`` on escape.
    """
    real = Path(os.path.realpath(path))
    try:
        real.relative_to(vault_root())
    except ValueError as exc:
        raise ValueError('Access denied: path escapes the vault boundary.') from exc
    return real


def resolve_in_vault(relative: str = '') -> Path | None:
    """Resolve a vault-relative path; ``None`` if it escapes the jail."""
    if reject_parent_segments(relative):
        return None
    root = vault_root()
    raw = relative.strip().replace('\\', '/').strip('/')
    candidate = Path(os.path.realpath(root / raw)) if raw else root
    try:
        assert_in_vault(candidate)
    except ValueError:
        return None
    return candidate
