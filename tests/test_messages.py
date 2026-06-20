"""Tests for brain.core.messages helpers."""

from __future__ import annotations

from brain.core.messages import parse_tool_arguments


class TestParseToolArguments:
    def test_dict_passthrough(self):
        assert parse_tool_arguments({'a': 1}) == {'a': 1}

    def test_json_string(self):
        assert parse_tool_arguments('{"note_title": "x"}') == {'note_title': 'x'}

    def test_invalid_json_returns_empty(self):
        assert parse_tool_arguments('{bad') == {}

    def test_empty_string(self):
        assert parse_tool_arguments('') == {}
