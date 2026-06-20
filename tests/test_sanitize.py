"""Tests for brain.core.sanitize text cleanup pipeline."""

from __future__ import annotations

from rich.cells import cell_len

from brain.core.context import ApplicationContext
from brain.core.defaults import APP_DEFAULTS
from brain.core.sanitize import (
    approximate_latex_display,
    extract_display_thinking,
    hard_wrap_line,
    normalize_markdown_rulers,
    prewrap_for_terminal,
    sanitize_assistant_text,
)


class TestSanitizeAssistantText:
    def test_strips_emoji_vs16_for_panel_width(self, ctx: ApplicationContext) -> None:
        """U+FE0F after pictograph: drop so Rich vs terminal width stay aligned."""
        raw = '🖼\uFE0F Breakdown by Image:\n\nHello.'
        out = sanitize_assistant_text(raw, ctx)
        assert '\uFE0F' not in out
        assert '🖼 Breakdown' in out


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


class TestExtractDisplayThinking:
    def test_strips_numbered_scaffold_and_keeps_last_prose(self):
        raw = """Here's a thinking process:

1.  **Analyze User Input:** The user asks to list colors.
2.  **Identify Constraints:** None besides three colors.
3.  **Formulate Response:** Red, Blue, Green.

Draft: Here are three colors: Red, Blue, Green.
Check against guidelines: None violated.

Self-Correction/Refinement during thought: The prompt is extremely simple, so I will just list them clearly.

Final Output Generation matches the draft.✅"""
        out = extract_display_thinking(raw)
        assert 'Analyze User Input' not in out
        assert 'Draft:' not in out
        assert 'extremely simple' in out

    def test_scaffold_only_returns_empty(self):
        raw = """Thinking Process:

1.  **Analyze User Input:** What is 2+2?
2.  **Calculate:** 2 + 2 = 4
3.  **Generate Response:** 4✅"""
        assert extract_display_thinking(raw) == ''

    def test_partial_scaffold_stream_returns_empty_until_prose(self):
        partial = "Here's a thinking process:\n\n1.  **Analyze User Input:** math"
        assert extract_display_thinking(partial) == ''
        complete = (
            partial + '\n2.  **Calculate:** 2+2=4\n\n'
            'The sum is four because addition combines both values.'
        )
        out = extract_display_thinking(complete)
        assert 'Analyze User Input' not in out
        assert 'addition combines both values' in out

    def test_single_paragraph_mixed_scaffold_keeps_last_prose_line(self):
        raw = (
            "1. Analyze the question\n"
            '2. Compute the answer\n'
            'Therefore the result is four.'
        )
        out = extract_display_thinking(raw)
        assert 'Analyze' not in out
        assert 'Therefore the result is four.' in out

    def test_bold_scaffold_steps_are_stripped(self):
        raw = (
            '**Analyze User Input:** list colors\n'
            '**Formulate Response:** red, blue, green\n\n'
            'I will answer with three common colors.'
        )
        out = extract_display_thinking(raw)
        assert 'Analyze User Input' not in out
        assert 'three common colors' in out

    def test_streaming_mode_hides_scaffold_until_substantive_tail(self):
        partial = (
            "Here's a thinking process:\n\n"
            '1.  **Analyze User Input:** math\n'
            '2.  **Calculate:** 2+2=4'
        )
        assert extract_display_thinking(partial, streaming=True) == ''
        with_tail = partial + '\n\nThe answer is four because 2+2 equals 4.'
        out = extract_display_thinking(with_tail, streaming=True)
        assert 'Analyze User Input' not in out
        assert 'answer is four' in out
