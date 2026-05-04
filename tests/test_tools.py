"""Tests for brain.tools vault operations."""

from __future__ import annotations

from brain.tools import (
    ToolResult,
    create_note,
    delete_note,
    edit_note,
    get_backlinks,
    list_directory,
    move_note,
    read_note,
    rename_note,
    search_by_tag,
    search_notes,
    update_frontmatter,
)


def _ok(r: ToolResult) -> str:
    assert r.success, f'Expected success, got: {r.message}'
    return r.message


def _err(r: ToolResult) -> str:
    assert not r.success, f'Expected failure, got: {r.message}'
    return r.message


class TestReadNote:
    def test_read_existing(self, ctx):
        msg = _ok(read_note('Note A'))
        assert '# Note A' in msg
        assert 'Hello world' in msg

    def test_read_by_path(self, ctx):
        msg = _ok(read_note('Subfolder/Deep Note'))
        assert '# Deep' in msg

    def test_read_missing(self, ctx):
        _err(read_note('Nonexistent'))

    def test_read_shows_outgoing_links(self, ctx):
        msg = _ok(read_note('Note B'))
        assert 'Note A' in msg
        assert 'Outgoing links' in msg


class TestCreateNote:
    def test_create_and_read(self, ctx, vault):
        _ok(create_note('New Note', '# Fresh\nContent here.'))
        assert (vault / 'New Note.md').is_file()
        assert '# Fresh' in _ok(read_note('New Note'))

    def test_create_in_subfolder(self, ctx, vault):
        _ok(create_note('Sub Note', 'body', folder='Subfolder'))
        assert (vault / 'Subfolder' / 'Sub Note.md').is_file()

    def test_create_duplicate(self, ctx):
        msg = _err(create_note('Note A', 'dup'))
        assert 'already exists' in msg


class TestEditNote:
    def test_append(self, ctx):
        _ok(edit_note('Note A', 'appended text', mode='append'))
        content = _ok(read_note('Note A'))
        assert 'Hello world' in content
        assert 'appended text' in content

    def test_prepend(self, ctx):
        _ok(edit_note('Note A', 'prepended text', mode='prepend'))
        content = _ok(read_note('Note A'))
        assert content.index('prepended text') < content.index('Hello world')

    def test_overwrite(self, ctx):
        _ok(edit_note('Note A', 'replaced', mode='overwrite'))
        content = _ok(read_note('Note A'))
        assert 'Hello world' not in content
        assert 'replaced' in content

    def test_invalid_mode(self, ctx):
        _err(edit_note('Note A', 'x', mode='invalid'))


class TestDeleteNote:
    def test_delete_requires_confirm(self, ctx):
        msg = _err(delete_note('Note A', confirm=False))
        assert 'confirmation' in msg

    def test_delete_with_confirm(self, ctx, vault):
        _ok(delete_note('Note A', confirm=True))
        assert not (vault / 'Note A.md').exists()


class TestMoveNote:
    def test_move_to_subfolder(self, ctx, vault):
        _ok(move_note('Note A', 'Subfolder'))
        assert (vault / 'Subfolder' / 'Note A.md').is_file()
        assert not (vault / 'Note A.md').exists()

    def test_move_conflict(self, ctx, vault):
        _ok(create_note('Deep Note', 'dup', folder=''))
        msg = _err(move_note('Deep Note', 'Subfolder'))
        assert 'already exists' in msg


class TestPathTraversal:
    def test_list_directory_escape(self, ctx):
        _err(list_directory('../../etc'))

    def test_create_note_escape(self, ctx):
        _err(create_note('evil', 'payload', folder='../../tmp'))


class TestSearchNotes:
    def test_basic_search(self, ctx):
        msg = _ok(search_notes('Hello'))
        assert 'Note A' in msg

    def test_search_no_results(self, ctx):
        msg = _ok(search_notes('zzzznonexistent'))
        assert 'No notes found' in msg

    def test_search_filename_match(self, ctx):
        msg = _ok(search_notes('Deep Note'))
        assert 'filename match' in msg

    def test_search_multiple_matches_per_file(self, ctx, vault):
        (vault / 'Multi.md').write_text(
            'line1 keyword\nline2\nline3 keyword\nline4\nline5 keyword\n',
            encoding='utf-8',
        )
        from brain.tools import _NOTE_INDEX
        _NOTE_INDEX.invalidate()
        msg = _ok(search_notes('keyword'))
        lines = [ln for ln in msg.splitlines() if 'keyword' in ln and 'Multi' in ln]
        assert len(lines) == 3


class TestGetBacklinks:
    def test_backlink_found(self, ctx):
        msg = _ok(get_backlinks('Note A'))
        assert 'Note B' in msg

    def test_no_backlinks(self, ctx):
        msg = _ok(get_backlinks('Deep Note'))
        assert 'No backlinks' in msg


class TestListDirectory:
    def test_list_root(self, ctx):
        msg = _ok(list_directory())
        assert 'Note A' in msg
        assert 'Subfolder' in msg

    def test_list_subfolder(self, ctx):
        msg = _ok(list_directory('Subfolder'))
        assert 'Deep Note' in msg


class TestUpdateFrontmatter:
    def test_update_existing(self, ctx):
        _ok(update_frontmatter('Note B', {'status': 'done'}))
        content = _ok(read_note('Note B'))
        assert 'done' in content

    def test_update_missing_note(self, ctx):
        _err(update_frontmatter('Nonexistent', {'x': 1}))


class TestRenameNote:
    def test_rename_in_place(self, ctx, vault):
        _ok(rename_note('Note A', 'Note Alpha'))
        assert not (vault / 'Note A.md').exists()
        assert (vault / 'Note Alpha.md').is_file()
        assert '# Note A' in _ok(read_note('Note Alpha'))

    def test_rename_conflict(self, ctx):
        msg = _err(rename_note('Note A', 'Note B'))
        assert 'already exists' in msg

    def test_rename_missing(self, ctx):
        _err(rename_note('Nonexistent', 'Whatever'))

    def test_rename_updates_wikilinks(self, ctx, vault):
        _ok(rename_note('Note A', 'Note Alpha'))
        content = (vault / 'Note B.md').read_text(encoding='utf-8')
        assert '[[Note Alpha]]' in content
        assert '[[Note A]]' not in content


class TestSearchByTag:
    def test_finds_tagged_note(self, ctx):
        msg = _ok(search_by_tag('python'))
        assert 'Note B' in msg

    def test_case_insensitive(self, ctx):
        msg = _ok(search_by_tag('PYTHON'))
        assert 'Note B' in msg

    def test_no_matches(self, ctx):
        msg = _ok(search_by_tag('nonexistent_tag'))
        assert 'No notes found' in msg
