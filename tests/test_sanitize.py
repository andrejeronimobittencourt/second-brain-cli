"""Tests for brain.sanitize text cleanup pipeline."""

from __future__ import annotations

from rich.cells import cell_len

from brain.defaults import APP_DEFAULTS
from brain.sanitize import (
    approximate_latex_display,
    hard_wrap_line,
    normalize_markdown_rulers,
    prewrap_for_terminal,
)


class TestNormalizeMarkdownRulers:
    def test_long_dashes_collapsed(self):
        text = '-' * 50
        assert normalize_markdown_rulers(text, APP_DEFAULTS) == '---'

    def test_short_dashes_kept(self):
        text = '---'
        assert normalize_markdown_rulers(text, APP_DEFAULTS) == '---'

    def test_normal_text_unchanged(self):
        text = 'Hello world\nSecond line'
        assert normalize_markdown_rulers(text, APP_DEFAULTS) == text


class TestApproximateLatexDisplay:
    def test_arrow_replacement(self):
        pairs = APP_DEFAULTS.latex_symbol_pairs
        assert '→' in approximate_latex_display(r'$\rightarrow$', pairs)

    def test_alpha_replacement(self):
        pairs = APP_DEFAULTS.latex_symbol_pairs
        assert 'α' in approximate_latex_display(r'$\alpha$', pairs)

    def test_plain_text_unchanged(self):
        pairs = APP_DEFAULTS.latex_symbol_pairs
        text = 'No latex here'
        assert approximate_latex_display(text, pairs) == text


class TestHardWrapLine:
    def test_short_line_unchanged(self):
        line = 'short'
        assert hard_wrap_line(line, 80, APP_DEFAULTS) == 'short'

    def test_long_line_wrapped(self):
        line = 'word ' * 30
        result = hard_wrap_line(line.strip(), 40, APP_DEFAULTS)
        assert '\n' in result
        # Joiner is GFM hard break ``'  \\n'``; split on that to recover wrapped chunks.
        for chunk in result.split('  \n'):
            assert cell_len(chunk) <= 40

    def test_emoji_counts_as_two_cells(self):
        """Leading emoji must not push the first line past the cell budget."""
        line = '💡 ' + ('word ' * 25)
        result = hard_wrap_line(line, 40, APP_DEFAULTS)
        first = result.split('\n')[0]
        assert cell_len(first) <= 40


class TestPrewrapForTerminal:
    def test_table_lines_not_split(self):
        body = '| a | b |\n| --- | --- |'
        out = prewrap_for_terminal(body, 10, APP_DEFAULTS)
        assert out == body

    def test_prose_lines_use_gfm_hard_break(self):
        out = prewrap_for_terminal('first line\nsecond line', 80, APP_DEFAULTS)
        assert out == 'first line  \nsecond line'

    def test_code_fence_uses_plain_newlines(self):
        body = 'before\n```py\nx = 1\n```\nafter'
        out = prewrap_for_terminal(body, 80, APP_DEFAULTS)
        assert '```py\nx = 1\n```' in out
        assert 'before  \n```py' not in out

    def test_paragraph_blank_line_not_hard_break(self):
        out = prewrap_for_terminal('first\n\nsecond', 80, APP_DEFAULTS)
        assert out == 'first\n\nsecond'
