"""Tests for brain.core.tool_validate."""

from __future__ import annotations

from brain.core.tool_validate import validate_tool_call


class TestToolValidate:
    def test_rejects_dotdot_path(self):
        err = validate_tool_call('create_folder', {'relative_path': '../evil'})
        assert err is not None
        assert '..' in err

    def test_rejects_empty_note_title(self):
        err = validate_tool_call('read_note', {'note_title': '  '})
        assert err is not None

    def test_delete_without_confirm(self):
        err = validate_tool_call('delete_note', {'note_title': 'X', 'confirm': False})
        assert err is not None
        assert 'confirm' in err

    def test_delete_with_string_confirm(self):
        err = validate_tool_call('delete_note', {'note_title': 'X', 'confirm': 'true'})
        assert err is not None

    def test_read_note_slice_bounds(self):
        assert validate_tool_call('read_note', {'note_title': 'A', 'start_line': 0}) is not None
        assert validate_tool_call('read_note', {'note_title': 'A', 'max_lines': -1}) is not None
        assert validate_tool_call('read_note', {'note_title': 'A', 'start_line': 1, 'max_lines': 10}) is None

    def test_update_frontmatter_requires_dict(self):
        err = validate_tool_call(
            'update_frontmatter',
            {'note_title': 'A', 'metadata': 'tags'},
        )
        assert err is not None
