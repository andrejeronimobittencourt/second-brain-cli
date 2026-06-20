"""Rich (or plain) terminal UI bound to ``ApplicationContext``."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import textwrap
import threading
from collections.abc import Callable
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)

from brain.core.context import get_context
from brain.core.defaults import AppDefaults
from brain.core.sanitize import prewrap_for_terminal
from brain.core.streaming import merge_stream_content
from brain.repl.display_port import ReplDisplayPort, TurnPhase, active_display_port
from brain.vault.tools import CONTENT_HEAVY_TOOLS

_SPINNER_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

# Rich may emit cursor-motion codes that break prompt_toolkit FormattedTextControl.
# Do NOT match SGR color codes (\x1b[…m) — a character class like [Klm] would strip them.
_ANSI_CURSOR_MOTION_RE = re.compile(
    r'\x1b\[[0-9;]*[HfABCDEFGJKSTsu]|'
    r'\x1b\[[0-9;]*K|'
    r'\x1b\].*?(?:\x07|\x1b\\)|'
    r'\x1b\?[0-9;]*[hl]|'
    r'\r',
)

_ANSI_SGR_RE = re.compile(r'\x1b\[[0-9;]*m')


def _repl_display_sink() -> Any:
    """Return the active REPL display sink, if any."""
    return get_context().repl_display


def _active_console() -> Any:
    """
    Return the Rich console wired to the current ``sys.stdout``.

    Outside the REPL this is the process TTY; inside the REPL, Rich markup is
    routed through :class:`~brain.repl.display.ReplDisplaySink` instead.
    """
    ctx = get_context()
    con = ctx.console
    if con is not None:
        con.file = sys.stdout
    return con

try:
    from rich.constrain import Constrain
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Live = None  # type: ignore[misc, assignment]

try:
    from rich.status import Status as RichStatus

    RICH_STATUS_AVAILABLE = True
except ImportError:
    RichStatus = None  # type: ignore[misc, assignment]
    RICH_STATUS_AVAILABLE = False


def _strip_ansi_cursor_motion(text: str) -> str:
    """Remove cursor-movement escapes unsuitable for the REPL transcript pane."""
    if not text:
        return ''
    return _ANSI_CURSOR_MOTION_RE.sub('', text)


def _plain_from_ansi(text: str) -> str:
    """Best-effort visible text when prompt_toolkit cannot parse Rich ANSI."""
    if not text:
        return ''
    stripped = _ANSI_SGR_RE.sub('', _strip_ansi_cursor_motion(text))
    return stripped


def _rich_console_for_capture(*, width: int | None = None) -> Any | None:
    """Build a Rich console that renders into memory instead of the TTY."""
    ctx = get_context()
    if not RICH_AVAILABLE or ctx.console is None:
        return None
    from rich.console import Console
    from rich.theme import Theme

    theme = Theme(dict(ctx.defaults.ui.rich_theme))
    color_system = ctx.console.color_system or 'standard'
    return Console(
        file=StringIO(),
        force_terminal=True,
        color_system=color_system,
        width=width or _panel_outer_width(),
        legacy_windows=False,
        theme=theme,
    )


def _capture_rich_render(render_fn: Callable[[Any], None]) -> str:
    """Render via Rich into a captured ANSI string."""
    con = _rich_console_for_capture()
    if con is None:
        return ''
    render_fn(con)
    captured = con.file.getvalue()
    if not captured:
        return ''
    stripped = captured.rstrip('\n')
    cleaned = _strip_ansi_cursor_motion(stripped)
    visible = cleaned if cleaned.strip() else _plain_from_ansi(stripped)
    if not visible.strip():
        return ''
    return visible + '\n'


def plain_spinner_line(message: str, frame: int) -> str:
    """Plain spinner row when Rich capture is unavailable."""
    glyph = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    return f'{glyph} {message}\n'


def plain_answer_block(content: str) -> str:
    """Plain assistant panel block for the REPL transcript fallback."""
    title = get_context().defaults.ui.panel_answer_title
    return f'\n{title}:\n{content}\n'


def plain_thinking_block(content: str) -> str:
    """Plain reasoning panel block for the REPL transcript fallback."""
    title = get_context().defaults.ui.panel_reasoning_title
    return f'\n{title}:\n{content}\n'


def render_markup_to_ansi(markup: str) -> str:
    """Render one Rich markup string to ANSI (no cursor-movement escapes)."""
    if not markup:
        return ''

    def _render(con: Any) -> None:
        con.print(markup, soft_wrap=True, highlight=False)

    return _capture_rich_render(_render)


def render_spinner_to_ansi(message: str, frame: int) -> str:
    """Render one spinner frame for the REPL output pane."""
    glyph = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    ansi = render_markup_to_ansi(
        f'[agent.spinner]{glyph}[/] [agent.info]{message}[/]',
    )
    if ansi:
        return ansi if ansi.endswith('\n') else f'{ansi}\n'
    return plain_spinner_line(message, frame)


def _content_panel(
    content: str,
    *,
    title_rich_markup: str,
    border_style: str,
    markdown: bool,
) -> Any:
    """Build a content-height Rich panel (plain or markdown body)."""
    d, outer, inner = _markdown_panel_layout()
    body_text = content if content.strip() else ' '
    wrapped = prewrap_for_terminal(body_text, inner, d)
    body: Any = (
        _constrained_markdown(wrapped, inner)
        if markdown and content.strip()
        else wrapped
    )
    return Panel(
        body,
        title=title_rich_markup,
        border_style=border_style,
        width=outer,
        padding=(0, 1),
        expand=False,
    )


def render_answer_panel_to_ansi(content: str, *, markdown: bool) -> str:
    """Render the assistant answer panel to a static ANSI block."""
    if not RICH_AVAILABLE:
        title = get_context().defaults.ui.panel_answer_title
        return f'\n{title}:\n{content}\n'
    title = get_context().defaults.ui.panel_answer_title

    def _render(con: Any) -> None:
        t = get_context().defaults.terminal
        con.print(
            _content_panel(
                content,
                title_rich_markup=f'[agent.name]{title}[/]',
                border_style='cyan',
                markdown=markdown,
            ),
            crop=t.print_crop,
            overflow=t.print_overflow,
            soft_wrap=False,
        )

    return _capture_rich_render(_render) or plain_answer_block(content)


def render_info_panel_to_ansi(content: str) -> str:
    """Render a dim bordered panel for CLI status messages (``/stats``, etc.)."""
    if not (content or '').strip():
        return ''
    if not RICH_AVAILABLE:
        return f'{content.strip()}\n'

    def _render(con: Any) -> None:
        term = get_context().defaults.terminal
        d, outer, inner = _markdown_panel_layout()
        wrapped = prewrap_for_terminal(content.strip(), inner, d)
        con.print(
            Panel(
                f'[agent.info]{wrapped}[/]',
                border_style='dim',
                width=outer,
                padding=(0, 1),
                expand=False,
            ),
            crop=term.print_crop,
            overflow=term.print_overflow,
            soft_wrap=False,
        )

    return _capture_rich_render(_render)


def render_thinking_panel_to_ansi(content: str, *, markdown: bool = False) -> str:
    """Render the reasoning panel to a static ANSI block."""
    if not (content or '').strip():
        return ''
    if not RICH_AVAILABLE:
        title = get_context().defaults.ui.panel_reasoning_title
        return f'\n[{title}]\n{content}\n'
    title = get_context().defaults.ui.panel_reasoning_title

    def _render(con: Any) -> None:
        term = get_context().defaults.terminal
        con.print(
            _content_panel(
                content,
                title_rich_markup=f'[agent.info]{title}[/]',
                border_style='dim',
                markdown=markdown,
            ),
            crop=term.print_crop,
            overflow=term.print_overflow,
            soft_wrap=False,
        )

    return _capture_rich_render(_render) or plain_thinking_block(content)


class _PlainLineSpinner:
    """Minimal TTY line animation when Rich ``Status`` is unavailable."""

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._frames = '|/-\\'
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            ch = self._frames[i % len(self._frames)]
            sys.stdout.write(f'\r{self._message} {ch}\x1b[K')
            sys.stdout.flush()
            if self._stop.wait(0.1):
                break
            i += 1

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        sys.stdout.write('\r\x1b[K')
        sys.stdout.flush()


class GenerationWaitHandle:
    """
    Transient spinner until the first streamed model activity.

    Call :meth:`finish` once content, thinking, or tool calls begin.
    :meth:`finish` is idempotent and safe from a ``finally`` block.
    """

    def __init__(self, message: str, spinner_name: str) -> None:
        self._finished = False
        self._status: Any = None
        self._plain: _PlainLineSpinner | None = None
        self._repl_sink = _repl_display_sink()
        if self._repl_sink is not None:
            self._repl_sink.start_spinner(message, spinner_name)
            return
        ctx = get_context()
        con = _active_console()
        if (
            RICH_AVAILABLE
            and RICH_STATUS_AVAILABLE
            and RichStatus is not None
            and con is not None
        ):
            try:
                self._status = RichStatus(
                    f'[agent.info]{message}[/]',
                    console=con,
                    spinner=spinner_name,
                )
                self._status.start()
            except Exception:
                logger.debug('Rich Status spinner unavailable; falling back to plain spinner', exc_info=True)
                self._status = None
        if self._status is None and sys.stdout.isatty():
            self._plain = _PlainLineSpinner(message)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._repl_sink is not None:
            self._repl_sink.stop_spinner()
            return
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self._plain is not None:
            self._plain.stop()
            self._plain = None


def generation_wait_start(message: str) -> GenerationWaitHandle:
    """
    Show a loading line until :meth:`GenerationWaitHandle.finish` runs.

    Uses Rich ``Status`` when a Rich console is configured; otherwise a plain
    TTY spinner. Does nothing visible when stdout is not a TTY and Rich status
    is unavailable.
    """
    ctx = get_context()
    spinner = ctx.defaults.terminal.cli_generation_spinner
    return GenerationWaitHandle(message, spinner)


def _panel_outer_width() -> int:
    ctx = get_context()
    d = ctx.defaults.terminal
    pad = d.panel_width_safety_columns
    if ctx.repl_display is not None:
        if ctx.repl_terminal_columns > 0:
            return max(d.panel_min_outer_width, ctx.repl_terminal_columns - pad)
        return d.panel_fallback_outer_width
    candidates: list[int] = []
    for fd in (sys.__stdout__, sys.stdout):
        try:
            if fd is not None and hasattr(fd, 'fileno'):
                candidates.append(shutil.get_terminal_size(fd.fileno()).columns)
                break
        except (OSError, AttributeError, ValueError):
            continue
    try:
        candidates.append(shutil.get_terminal_size().columns)
    except OSError:
        pass
    env_c = os.environ.get('COLUMNS', '').strip()
    if env_c.isdigit():
        candidates.append(int(env_c))
    con = ctx.console
    if con is not None:
        cw = con.width
        if cw is not None:
            candidates.append(cw)
    if not candidates:
        return d.panel_fallback_outer_width
    raw = min(candidates)
    pad = d.panel_width_safety_columns
    return max(d.panel_min_outer_width, raw - pad)


def repl_transcript_inner_width() -> int:
    """Usable text width for REPL transcript lines (prompts, wrapped prose)."""
    d = get_context().defaults
    outer = _panel_outer_width()
    margin = d.terminal.panel_inner_margin_columns
    shrink = d.terminal.panel_prewrap_shrink_columns
    min_inner = d.terminal.prewrap_min_inner_width
    return max(min_inner, outer - margin - shrink)


def _constrained_markdown(wrapped: str, inner: int) -> Any:
    """
    Wrap Markdown so Rich passes a strict max width into layout.

    Without this, Panel children may see a wider ``options.max_width`` than the
    width used by ``prewrap_for_terminal``, and Markdown (lists, paragraphs) can
    reflow past the visible panel and look truncated at the terminal edge.
    """
    w = max(1, inner)
    return Constrain(Markdown(wrapped), width=w)


def _markdown_panel_layout() -> tuple[AppDefaults, int, int]:
    """Return ``(defaults, outer_width, inner_width)`` for markdown panels."""
    d = get_context().defaults
    outer = _panel_outer_width()
    margin = d.terminal.panel_inner_margin_columns
    shrink = d.terminal.panel_prewrap_shrink_columns
    min_inner = d.terminal.prewrap_min_inner_width
    inner = max(min_inner, outer - margin - shrink)
    return d, outer, inner


def _render_markdown_panel(
    content: str,
    *,
    title_rich_markup: str,
    border_style: str,
    plain_header: str,
) -> None:
    """
    Shared Rich/plain path for assistant answer and thinking panels.

    *plain_header* is printed before *content* on the plain path (for example
    ``'\\nAnswer:\\n'`` or ``'\\n[Thinking]\\n'``).
    """
    ctx = get_context()
    port = active_display_port()
    if isinstance(port, ReplDisplayPort):
        if border_style == 'cyan' and (port.answer_is_active() or port.has_live_answer()):
            port.end_answer(final_content=content)
            return
        if border_style == 'cyan':
            port.print_agent_panel(content)
        else:
            port.print_thinking_panel(content)
        return
    if not RICH_AVAILABLE or ctx.console is None:
        print(f'{plain_header}{content}')
        return
    d, outer, inner = _markdown_panel_layout()
    wrapped = prewrap_for_terminal(content, inner, d)
    _print_panel_rich(
        Panel(
            _constrained_markdown(wrapped, inner),
            title=title_rich_markup,
            border_style=border_style,
            width=outer,
            padding=(0, 1),
            expand=False,
        ),
    )


def _print_panel_rich(panel: Any) -> None:
    con = _active_console()
    if con is None:
        return
    ctx = get_context()
    t = ctx.defaults.terminal
    con.print(
        panel,
        crop=t.print_crop,
        overflow=t.print_overflow,
        soft_wrap=False,
    )


def _render_agent_panel(content: str) -> None:
    """Render a pre-sanitized assistant response panel."""
    title = get_context().defaults.ui.panel_answer_title
    _render_markdown_panel(
        content,
        title_rich_markup=f'[agent.name]{title}[/]',
        border_style='cyan',
        plain_header=f'\n{title}:\n',
    )


def print_agent(content: str) -> None:
    """
    Render a pre-sanitized assistant response to the terminal.

    Content is expected to have been processed by ``sanitize_assistant_text``
    before being passed here; this function only handles layout and display.
    """
    port = active_display_port()
    if isinstance(port, ReplDisplayPort):
        port.print_agent_panel(content)
        return
    _render_agent_panel(content)


def preview_tool_args_for_terminal(
    tool_name: str,
    args: dict[str, Any],
    *,
    content_preview_chars: int,
) -> dict[str, Any]:
    """
    Build a shallow copy of tool *args* suitable for terminal logging.

    ``create_note`` / ``edit_note`` show note body size only (no text preview).
    The file on disk is unchanged — only the CLI preview is shortened.
    """
    _ = content_preview_chars
    if tool_name not in CONTENT_HEAVY_TOOLS:
        return dict(args)
    out = dict(args)
    raw = out.get('content')
    if isinstance(raw, str) and raw:
        out['content'] = f'{len(raw)} chars'
    return out


def _format_tool_arg_value(value: Any, *, max_chars: int) -> str:
    """Render one argument value for the terminal, truncating long ``repr`` strings."""
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return f'{text[: max(1, max_chars - 1)]}…'


def format_tool_call_line(
    name: str,
    args: dict[str, Any],
    *,
    width: int,
    max_arg_chars: int,
    content_preview_chars: int,
) -> str:
    """Return a plain tool-call line wrapped to *width* columns."""
    shown = preview_tool_args_for_terminal(
        name,
        args,
        content_preview_chars=content_preview_chars,
    )
    arg_parts = [
        f'{k}={_format_tool_arg_value(v, max_chars=max_arg_chars)}'
        for k, v in shown.items()
    ]
    inner = f'{name}({", ".join(arg_parts)})'
    prefix = '  ⚙  '
    usable = max(24, width)
    if len(prefix) + len(inner) <= usable:
        return f'{prefix}{inner}'
    return textwrap.fill(
        inner,
        width=max(16, usable - len(prefix)),
        initial_indent=prefix,
        subsequent_indent=' ' * len(prefix),
        break_long_words=True,
        break_on_hyphens=False,
    )


def print_tool(name: str, args: dict[str, Any]) -> None:
    ctx = get_context()
    d: AppDefaults = ctx.defaults
    plain = format_tool_call_line(
        name,
        args,
        width=repl_transcript_inner_width(),
        max_arg_chars=d.limits.max_tool_arg_chars,
        content_preview_chars=d.limits.tool_call_content_preview_chars,
    )
    line = f'[agent.tool]{plain}[/]'
    sink = _repl_display_sink()
    if sink is not None:
        sink.append_rich(line)
        return
    if ctx.console:
        ctx.console.print(f'[agent.tool]{plain}[/]')
    else:
        print(plain)


def print_error(msg: str) -> None:
    sink = _repl_display_sink()
    if sink is not None:
        sink.append_rich(f'[agent.error]✗  {msg}[/]')
        return
    ctx = get_context()
    if ctx.console:
        ctx.console.print(f'[agent.error]✗  {msg}[/]')
    else:
        print(f'ERROR: {msg}')


def print_info(msg: str) -> None:
    sink = _repl_display_sink()
    if sink is not None:
        sink.append_info_panel(msg)
        return
    ctx = get_context()
    if ctx.console:
        ctx.console.print(f'[agent.info]{msg}[/]')
    else:
        print(msg)


def print_terminal_info(msg: str) -> None:
    """Write an info line to the real TTY (for example after the REPL UI exits)."""
    text = msg.strip()
    if not text:
        return
    out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    ctx = get_context()
    if RICH_AVAILABLE and ctx.console is not None:
        from rich.console import Console
        from rich.theme import Theme

        con = Console(
            file=out,
            force_terminal=True,
            color_system=ctx.console.color_system or 'standard',
            theme=Theme(dict(ctx.defaults.ui.rich_theme)),
        )
        con.print(f'[agent.info]{text}[/]')
        return
    print(text, file=out)


def _render_thinking_panel(content: str) -> None:
    """Render filtered reasoning to the terminal."""
    t = get_context().defaults.ui.panel_reasoning_title
    _render_markdown_panel(
        content,
        title_rich_markup=f'[agent.info]{t}[/]',
        border_style='dim',
        plain_header=f'\n[{t}]\n',
    )


def print_thinking(content: str) -> None:
    """
    Render filtered model reasoning to the terminal.

    Applies ``extract_display_thinking`` as a safety net before layout.
    """
    if not (content or '').strip():
        return
    from brain.core.sanitize import extract_display_thinking, sanitize_assistant_text

    ctx = get_context()
    extracted = extract_display_thinking(content, streaming=False)
    if not extracted:
        return
    text = sanitize_assistant_text(extracted, ctx)
    port = active_display_port()
    if isinstance(port, ReplDisplayPort):
        port.print_thinking_panel(text)
        return
    _render_thinking_panel(text)


def _answer_panel(content: str, *, markdown: bool) -> Any:
    """Build the assistant answer panel for streaming and final render."""
    title = get_context().defaults.ui.panel_answer_title
    return _content_panel(
        content,
        title_rich_markup=f'[agent.name]{title}[/]',
        border_style='cyan',
        markdown=markdown,
    )


def cancel_answer_stream() -> None:
    """Abort the active answer stream without committing a panel."""
    active_display_port().cancel_answer()


def begin_answer_stream(*, plain: bool = False) -> None:
    """Start streaming an assistant reply (Live panel or plain header)."""
    active_display_port().begin_answer(plain=plain)


def write_answer_delta(text: str) -> None:
    """Append one streamed token chunk to the active answer stream."""
    active_display_port().write_answer_delta(text)


def end_answer_stream(*, final_content: str | None = None) -> None:
    """
    Finish the active answer stream.

    REPL panels stay plain (pre-wrapped) on finalize so height matches the
    live stream and does not expand when Rich Markdown re-renders the body.
    """
    active_display_port().end_answer(final_content=final_content)


def answer_stream_is_active() -> bool:
    """Return True while an answer stream is in progress."""
    return active_display_port().answer_is_active()


def thinking_stream_is_active() -> bool:
    """Return True while a reasoning stream is in progress."""
    return active_display_port().thinking_is_active()


def cancel_thinking_stream() -> None:
    """Abort the active reasoning stream without committing a panel."""
    active_display_port().cancel_thinking()


def begin_thinking_stream() -> None:
    """Start streaming model reasoning into the active display port."""
    port = active_display_port()
    port.cancel_thinking()
    if isinstance(port, ReplDisplayPort):
        port.thinking.active = True
        port.sink.begin_thinking_stream()
        port.turn_phase = TurnPhase.STREAMING


def write_thinking_delta(text: str) -> None:
    """Append filtered display text to the active thinking stream (legacy hook)."""
    port = active_display_port()
    if isinstance(port, ReplDisplayPort):
        if not port.thinking.active:
            port.thinking.active = True
            port.sink.begin_thinking_stream()
            port.turn_phase = TurnPhase.STREAMING
        port.thinking.raw = merge_stream_content(port.thinking.raw, text)
        port.thinking.display = merge_stream_content(port.thinking.display, text)
        if port.thinking.display.strip():
            port._push_thinking_live(port.thinking.display, raw=port.thinking.raw)
        return
    if text:
        port.sync_thinking_display(text, displayed_prefix='')


def reveal_thinking_before_answer(raw_thinking: str) -> None:
    """
    Progressively show filtered reasoning buffered during a post-tool model hop.

    Called when answer content begins so tool-planning hops never paint a panel.
    """
    active_display_port().reveal_thinking_before_answer(raw_thinking)


def sync_thinking_display_stream(raw_thinking: str, *, displayed_prefix: str) -> str:
    """Stream display-ready reasoning; returns the prefix already shown."""
    return active_display_port().sync_thinking_display(
        raw_thinking,
        displayed_prefix=displayed_prefix,
    )


def end_thinking_stream(*, final_content: str | None = None) -> None:
    """Finish the active reasoning stream and commit the panel."""
    port = active_display_port()
    thinking = getattr(port, 'thinking', None)
    raw = thinking.raw if thinking is not None else ''
    port.commit_thinking(raw, final_content=final_content)
