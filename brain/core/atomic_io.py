"""Atomic text file writes shared by session and search index."""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path, payload: str, *, encoding: str = 'utf-8') -> None:
    """Write *payload* to *path* via a same-directory temp file and ``replace()``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        tmp.write_text(payload, encoding=encoding)
        tmp.replace(path)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
