"""Assistant output cleanup — uses ``AppDefaults`` + ``ApplicationContext`` ."""

from __future__ import annotations

import re

from rich.cells import cell_len

from brain.core.context import ApplicationContext
from brain.core.defaults import AppDefaults

_INLINE_MATH_DOLLAR = re.compile(r'\$(?P<body>[^$\n]+?)\$')
_TEXT_BRACE = re.compile(r'\\text\{([^}]*)\}')

# Model "thinking" often includes scaffold the UI must not show (numbered steps,
# meta headers, draft/checklist tails). Keep only concluding prose reasoning.
_THINKING_META_HEADER = re.compile(
    r"^(?:here(?:'s| is) a )?thinking process:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_STEP_LINE = re.compile(
    r'^\s*(?:'
    r'(?:\*\*)?\d+[.)]\s*'  # ``1.`` or ``**1.``
    r'|[-*•]\s+(?:\*\*)?'  # bullet steps
    r')',
)
_BOLD_SCAFFOLD_STEP = re.compile(
    r'^\*\*(?:Analyze|Identify|Formulate|Determine|Check|Verify|Evaluate|Plan|'
    r'Consider|Review|Parse|Extract|Search|Select|Confirm|Generate|Draft|Output|'
    r'Final|Calculate|Compute|Summarize|Respond|Provide|List|Draft)\b',
    re.IGNORECASE,
)
_THINKING_SCAFFOLD_HEADER = re.compile(
    r"(?:"
    r"here(?:'s| is) a (?:step-by-step )?(?:analysis|reasoning)(?:\s+for|\s+to|\s*:)?"
    r"|step-by-step(?:\s+(?:analysis|reasoning|process))?\s*:?"
    r")\s*$",
    re.IGNORECASE,
)
_META_TAIL_LINE = re.compile(
    r'^(?:draft:|check against|verify constraints|format output|'
    r'generate response|output generation|final output generation)',
    re.IGNORECASE,
)
_FINAL_DRAFT_TAIL = re.compile(r'matches the draft\.?\s*✅?\s*$', re.IGNORECASE)

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


def _is_scaffold_header_line(line: str) -> bool:
    """True for meta lines that introduce internal reasoning scaffold."""
    stripped = line.strip()
    if not stripped:
        return False
    if _THINKING_META_HEADER.match(stripped):
        return True
    return bool(_THINKING_SCAFFOLD_HEADER.match(stripped))


def _is_numbered_scaffold_line(line: str) -> bool:
    """True when a single line is a numbered analysis step (model scaffold)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _is_scaffold_header_line(stripped):
        return True
    if _BOLD_SCAFFOLD_STEP.match(stripped):
        return True
    return bool(_NUMBERED_STEP_LINE.match(stripped))


def _strip_scaffold_lines(text: str) -> str:
    """Remove numbered/meta scaffold lines; keep substantive prose only."""
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _is_numbered_scaffold_line(line):
            continue
        if _META_TAIL_LINE.match(line):
            continue
        kept.append(raw.rstrip())
    return '\n'.join(kept).strip()


def _last_scaffold_line_index(lines: list[str]) -> int:
    """Index of the last scaffold line, or -1."""
    last = -1
    for idx, raw in enumerate(lines):
        if _is_numbered_scaffold_line(raw):
            last = idx
    return last


def _is_numbered_scaffold_paragraph(para: str) -> bool:
    """True when every non-empty line is a numbered list step (model scaffold)."""
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_is_numbered_scaffold_line(ln) for ln in lines)


def _last_substantive_line_index(lines: list[str]) -> int:
    """Index of the last non-scaffold, non-meta-tail line, or -1."""
    last = -1
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if _is_numbered_scaffold_line(line):
            continue
        if _META_TAIL_LINE.match(line):
            continue
        last = idx
    return last


def _is_meta_tail_paragraph(para: str) -> bool:
    """True for internal checklist / draft lines, not user-facing reasoning."""
    first = para.strip().split('\n', 1)[0].strip()
    if _META_TAIL_LINE.match(first):
        return True
    return bool(_FINAL_DRAFT_TAIL.search(first))


def extract_display_thinking(text: str, *, streaming: bool = False) -> str:
    """
    Return the last substantive reasoning block for display.

    Strips common model scaffolding (meta headers, numbered analysis steps,
    draft/checklist tails) and keeps the final natural-language reasoning that
    precedes the assistant answer.

    When *streaming* is True, returns empty until scaffold markers are complete
    and a substantive tail exists — avoids leaking partial scaffold to the UI.
    """
    if not text or not text.strip():
        return ''
    cleaned = _THINKING_META_HEADER.sub('', text.strip()).strip()
    if not cleaned:
        return ''

    lines = cleaned.splitlines()
    has_scaffold = any(_is_numbered_scaffold_line(ln) for ln in lines)
    last_sub_idx = _last_substantive_line_index(lines)
    last_scaffold_idx = _last_scaffold_line_index(lines)

    if has_scaffold:
        if last_sub_idx < 0 or last_sub_idx <= last_scaffold_idx:
            return ''
        tail_lines = lines[last_sub_idx:]
        result = '\n'.join(tail_lines).strip()
        result = _strip_scaffold_lines(result)
        if streaming and not result:
            return ''
        return result if result and not _is_numbered_scaffold_paragraph(result) else ''

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', cleaned) if p.strip()]
    if not paragraphs:
        return ''

    last_para_scaffold_idx = -1
    for idx, para in enumerate(paragraphs):
        if _is_numbered_scaffold_paragraph(para):
            last_para_scaffold_idx = idx

    candidates = (
        paragraphs[last_para_scaffold_idx + 1 :]
        if last_para_scaffold_idx >= 0
        else paragraphs
    )
    substantive = [p for p in candidates if not _is_meta_tail_paragraph(p)]
    if substantive:
        result = substantive[-1].strip()
    elif last_para_scaffold_idx >= 0:
        return ''
    else:
        result = paragraphs[-1].strip()

    result = _strip_scaffold_lines(result)
    if not result:
        return ''
    if streaming and _is_numbered_scaffold_paragraph(result):
        return ''
    return result
