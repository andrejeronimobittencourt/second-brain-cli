"""Tests for brain.core.truncate."""

from __future__ import annotations

from brain.core.truncate import truncate_for_model


class TestTruncateForModel:
    def test_under_limits_unchanged(self):
        text = 'line one\nline two\n'
        assert truncate_for_model(text, max_lines=10, max_bytes=1000) == text

    def test_truncates_by_lines(self):
        text = '\n'.join(f'line {i}' for i in range(100))
        out = truncate_for_model(text, max_lines=5, max_bytes=100_000)
        assert 'line 0' in out
        assert 'truncated for model context' in out
        assert 'line 99' not in out

    def test_truncates_by_bytes(self):
        text = 'x' * 20_000
        out = truncate_for_model(text, max_lines=10_000, max_bytes=100)
        assert 'truncated for model context' in out
        assert len(out.encode('utf-8')) < 20_000
