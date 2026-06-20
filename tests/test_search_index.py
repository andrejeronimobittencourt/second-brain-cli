"""Tests for brain.vault.catalog search indexing."""

from __future__ import annotations

from brain.vault.catalog import INDEX_FILENAME, get_vault_catalog
from brain.vault.tools import create_note, edit_note, search_notes


class TestSearchIndex:
    def test_search_builds_index_file(self, ctx, vault):
        get_vault_catalog().invalidate()
        result = search_notes('Hello')
        assert result.success
        assert (vault / INDEX_FILENAME).is_file()

    def test_edit_updates_index(self, ctx, vault):
        get_vault_catalog().invalidate()
        search_notes('world')
        create_note('Indexed', 'unique keyword xyzzy')
        assert 'xyzzy' in search_notes('xyzzy').message
        edit_note('Indexed', '\nmore xyzzy text', mode='append')
        assert 'Indexed' in search_notes('more xyzzy').message
