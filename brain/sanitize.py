"""Assistant output cleanup — uses ``AppDefaults`` + ``ApplicationContext`` ."""

from __future__ import annotations

import re

from rich.cells import cell_len

from brain.context import ApplicationContext
from brain.defaults import AppDefaults

_INLINE_MATH_DOLLAR = re.compile(r'\$(?P<body>[^$\n]+?)\$')
_TEXT_BRACE = re.compile(r'\\text\{([^}]*)\}')

# U+FE0F (emoji presentation selector) after pictographs (e.g. U+1F5BC + VS16):
# Rich ``cell_len`` and some terminals disagree on whether VS16 adds width, so
# Markdown panel borders can drift by one column — strip before layout.
_VS16_EMOJI_PRESENTATION = '\uFE0F'


def _strip_emoji_presentation_vs16(text: str) -> str:
    """Drop VS16 so prewrap / Rich panels match common TTY display width."""
    return text.replace(_VS16_EMOJI_PRESENTATION, '')


def normalize_markdown_rulers(text: str, d: AppDefaults) -> str:
    """Collapse over-long HR lines so Rich panels stay aligned."""
    lines_out: list[str] = []
    min_rep = d.markdown.ruler_min_repeated_chars
    charset = frozenset(d.markdown.ruler_charset_chars)
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= min_rep and set(stripped) <= charset:
            lines_out.append('---')
        else:
            lines_out.append(line)
    return '\n'.join(lines_out)


def _latex_body_to_unicode(body: str, latex_pairs: tuple[tuple[str, str], ...]) -> str:
    s = body.strip()
    s = _TEXT_BRACE.sub(r'\1', s)
    for cmd, sym in latex_pairs:
        s = s.replace(cmd, sym)
    return s.strip()


def approximate_latex_display(text: str, latex_pairs: tuple[tuple[str, str], ...]) -> str:
    """Replace ``$...$`` with Unicode approximations."""

    def repl(m: re.Match[str]) -> str:
        inner = _latex_body_to_unicode(m.group('body'), latex_pairs)
        return inner if inner else m.group(0)

    return _INLINE_MATH_DOLLAR.sub(repl, text)


def _prefix_within_cell_budget(s: str, max_cells: int) -> int:
    """
    Largest character index ``i`` such that ``cell_len(s[:i]) <= max_cells``.

    Used so wrapping matches Rich / terminal display width (emoji, CJK, etc.).
    """
    if max_cells < 1 or not s:
        return 0
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cell_len(s[:mid]) <= max_cells:
            lo = mid
        else:
            hi = mid - 1
    return lo


def hard_wrap_line(line: str, max_len: int, d: AppDefaults) -> str:
    """
    Break long lines on spaces or hard-break.

    ``max_len`` is interpreted as **terminal display cells** (via Rich
    ``cell_len``), not Python string length, so emoji and wide characters do not
    spill past the panel and get clipped.
    """
    floor = d.terminal.hard_wrap_min_line_length
    if max_len < floor:
        max_len = floor
    if cell_len(line) <= max_len:
        return line
    frac = d.terminal.hard_wrap_space_break_min_fraction
    chunks: list[str] = []
    rest = line
    guard = 0
    while cell_len(rest) > max_len:
        guard += 1
        if guard > len(line) + 8:
            chunks.append(rest)
            break
        cut = _prefix_within_cell_budget(rest, max_len)
        if cut == 0:
            cut = 1
        window = rest[:cut]
        brk = window.rfind(' ')
        if brk > int(cut * frac):
            chunks.append(rest[:brk].rstrip())
            rest = rest[brk:].lstrip()
        else:
            chunks.append(window.rstrip())
            rest = rest[cut:].lstrip()
        if not rest:
            break
    if rest:
        chunks.append(rest)
    # GFM hard line break so Rich Markdown does not join segments with a space.
    if len(chunks) == 1:
        return chunks[0]
    return '  \n'.join(chunks)


def _is_markdown_table_line(stripped: str) -> bool:
    """True for GFM-style table rows (including ``| --- |`` separators)."""
    return stripped.startswith('|') and stripped.count('|') >= 2


def _is_fence_delimiter_line(stripped: str) -> bool:
    return stripped.startswith('```')


def _fence_depth_before_each_line(lines: list[str]) -> list[int]:
    """``depth_before[i]`` is fence nesting (0 or 1) before consuming ``lines[i]``."""
    depth = 0
    out: list[int] = []
    for line in lines:
        out.append(depth)
        if _is_fence_delimiter_line(line.lstrip()):
            depth ^= 1
    return out


def _join_prewrap_segments(
    source_lines: list[str],
    segments: list[str],
    depth_before: list[int],
) -> str:
    """
    Join pre-wrapped segments so Markdown does not reflow across line boundaries.

    Outside fenced code, consecutive lines use a GFM hard break (two spaces +
    newline). Across fences or at fence delimiters, a plain newline is used so
    code blocks stay valid. An empty source line becomes a Markdown paragraph
    break (blank line).
    """
    if not segments:
        return ''
    parts: list[str] = []
    idx = 0
    while idx < len(segments) and source_lines[idx] == '':
        idx += 1
    if idx > 0:
        parts.append('\n\n')
    if idx >= len(segments):
        return ''.join(parts)
    parts.append(segments[idx])
    for i in range(idx + 1, len(segments)):
        if source_lines[i] == '':
            if source_lines[i - 1] != '':
                parts.append('\n\n')
            continue
        prev_s = source_lines[i - 1].lstrip()
        cur_s = source_lines[i].lstrip()
        if source_lines[i - 1] == '':
            sep = ''
        elif (
            depth_before[i - 1] == 1
            or depth_before[i] == 1
            or _is_fence_delimiter_line(prev_s)
            or _is_fence_delimiter_line(cur_s)
            or _is_markdown_table_line(prev_s)
            or _is_markdown_table_line(cur_s)
        ):
            sep = '\n'
        else:
            sep = '  \n'
        parts.append(sep)
        parts.append(segments[i])
    return ''.join(parts)


def prewrap_for_terminal(text: str, inner_width: int, d: AppDefaults) -> str:
    """
    Fix up line-join separators before Rich Markdown renders.

    All source lines pass through verbatim — ``Constrain(Markdown, width=inner)``
    handles prose word-wrapping correctly.  Hard-wrapping prose here with GFM
    ``  \\n`` breaks is wrong: Rich Markdown ignores trailing-space line breaks
    and collapses them to spaces, causing long lines to overflow and be clipped.

    What this function *does* do is pick the right separator between adjacent
    lines when reassembling the text (``_join_prewrap_segments``): prose lines
    get ``  \\n`` (GFM hard-break between already-short lines), code-fence
    contents and table rows get plain ``\\n``, and blank lines between paragraphs
    become ``\\n\\n``.
    """
    floor = d.terminal.prewrap_min_inner_width
    if inner_width < floor:
        inner_width = floor
    source_lines = text.split('\n')
    depth_before = _fence_depth_before_each_line(source_lines)
    # Every line is passed through unchanged; separator logic lives in
    # _join_prewrap_segments.
    return _join_prewrap_segments(source_lines, source_lines, depth_before)


def sanitize_assistant_text(text: str, ctx: ApplicationContext) -> str:
    """Strip channel leaks, dedupe paragraphs, approximate LaTeX."""
    if not text:
        return text
    text = _strip_emoji_presentation_vs16(text)
    d = ctx.defaults
    text = normalize_markdown_rulers(text, d)
    text = ctx.channel_leak.sub('\n\n', text)
    chunks = re.split(r'\n{2,}', text.strip())
    out: list[str] = []
    prev_key = ''
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        key = re.sub(r'\s+', ' ', chunk)
        if key == prev_key:
            continue
        out.append(chunk)
        prev_key = key
    joined = '\n\n'.join(out)
    return approximate_latex_display(joined, ctx.latex_pairs)
