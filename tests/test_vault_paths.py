"""Tests for brain.vault.paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain.vault.paths import assert_in_vault, reject_parent_segments, resolve_in_vault


class TestRejectParentSegments:
    def test_rejects_dotdot(self):
        assert reject_parent_segments('../outside') is True
        assert reject_parent_segments('a/../b') is True

    def test_allows_normal(self):
        assert reject_parent_segments('Subfolder/Note') is False
        assert reject_parent_segments('') is False


@pytest.mark.skipif(
    os.name == 'nt' and not getattr(os, 'geteuid', None),
    reason='symlink tests may require elevated privileges on Windows',
)
class TestSymlinkJail:
    def test_symlink_outside_vault_rejected(self, ctx, vault: Path):
        # File must live outside the vault root (vault fixture == tmp_path).
        outside = vault.parent / 'outside_secret.txt'
        outside.write_text('secret', encoding='utf-8')
        link = vault / 'escape.md'
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip('symlinks not supported in this environment')
        assert resolve_in_vault('escape.md') is None
        with pytest.raises(ValueError, match='vault boundary'):
            assert_in_vault(link)
