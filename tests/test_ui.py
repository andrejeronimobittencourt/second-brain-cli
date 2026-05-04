"""Tests for ``brain.ui`` helpers."""

from __future__ import annotations

from brain.defaults import APP_DEFAULTS
from brain.ui import preview_tool_args_for_terminal


def test_preview_truncates_create_note_content():
    long = 'a' * 500
    out = preview_tool_args_for_terminal(
        'create_note',
        {'note_title': 'T', 'folder': 'F', 'content': long},
        content_preview_chars=100,
    )
    assert out['note_title'] == 'T'
    assert out['folder'] == 'F'
    assert len(out['content']) < len(long)
    assert '(500 chars)' in out['content']
    assert '\n' not in out['content']


def test_preview_multiline_content_shows_first_line():
    content = 'First line\nSecond line\nThird line'
    out = preview_tool_args_for_terminal(
        'create_note',
        {'note_title': 'T', 'content': content},
        content_preview_chars=80,
    )
    assert out['content'].startswith('First line')
    assert 'Second' not in out['content']
    assert f'({len(content)} chars)' in out['content']


def test_preview_short_singleline_passes_through():
    content = 'Short note.'
    out = preview_tool_args_for_terminal(
        'edit_note',
        {'note_title': 'T', 'content': content, 'mode': 'append'},
        content_preview_chars=80,
    )
    assert out['content'] == content


def test_preview_passes_through_other_tools():
    args = {'query': 'x'}
    assert preview_tool_args_for_terminal(
        'search_notes',
        args,
        content_preview_chars=APP_DEFAULTS.limits.tool_call_content_preview_chars,
    ) == args

