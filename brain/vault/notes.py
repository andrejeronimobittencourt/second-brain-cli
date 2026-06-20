"""Vault note iteration helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from brain.vault.paths import assert_in_vault, is_hidden_vault_path, vault_root


def iter_vault_notes() -> Iterator[Path]:
    """Yield sorted non-hidden ``.md`` notes inside the vault (real paths)."""
    root = vault_root()
    for abs_path in sorted(root.rglob('*.md')):
        try:
            rel = abs_path.relative_to(root)
        except ValueError:
            continue
        if is_hidden_vault_path(rel):
            continue
        try:
            yield assert_in_vault(abs_path)
        except ValueError:
            continue
