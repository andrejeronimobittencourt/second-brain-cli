"""Rich (or plain) terminal UI bound to ``ApplicationContext``."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

from brain.context import get_context
from brain.defaults import AppDefaults
from brain.sanitize import prewrap_for_terminal
from brain.tools import CONTENT_HEAVY_TOOLS

try:
    from rich.constrain import Constrain
    from rich.markdown import Markdown
    from rich.panel import Panel

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from rich.status import Status as RichStatus

    RICH_STATUS_AVAILABLE = True
except ImportError:
    RichStatus = None  # type: ignore[misc, assignment]
    RICH_STATUS_AVAILABLE = False


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
        ctx = get_context()
        con = ctx.console
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
    ctx = get_context()
    con = ctx.console
    if con is None:
        return
    t = ctx.defaults.terminal
    # Do not pass ``width=`` here: it can re-measure narrower than ``Panel`` and
    # crop the right edge while the border still fits the real terminal.
    # Pass soft_wrap=False explicitly: the Console is constructed with
    # soft_wrap=True (sensible default for plain-text output), but for panels
    # that is harmful — Rich translates soft_wrap=True to no_wrap=True, which
    # disables word-wrapping for the entire renderable tree (Markdown inside
    # Constrain inside Panel) and hard-clips text at the panel border.
    con.print(
        panel,
        crop=t.print_crop,
        overflow=t.print_overflow,
        soft_wrap=False,
    )


def print_agent(content: str) -> None:
    """
    Render a pre-sanitized assistant response to the terminal.

    Content is expected to have been processed by ``sanitize_assistant_text``
    before being passed here; this function only handles layout and display.
    """
    title = get_context().defaults.ui.panel_answer_title
    _render_markdown_panel(
        content,
        title_rich_markup=f'[agent.name]{title}[/]',
        border_style='cyan',
        plain_header=f'\n{title}:\n',
    )


def preview_tool_args_for_terminal(
    tool_name: str,
    args: dict[str, Any],
    *,
    content_preview_chars: int,
) -> dict[str, Any]:
    """
    Build a shallow copy of tool *args* suitable for one-line terminal logging.

    Long ``content`` for content-heavy tools is truncated; the file on disk is
    unchanged — only the CLI preview is shortened.
    """
    if tool_name not in CONTENT_HEAVY_TOOLS:
        return dict(args)
    out = dict(args)
    raw = out.get('content')
    if not isinstance(raw, str) or not raw:
        return out
    # Take the first non-empty line so the preview stays on one log line.
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), raw)
    total = len(raw)
    if len(first_line) <= content_preview_chars and len(raw) == len(first_line):
        return out  # short single-line content — show as-is
    snippet = first_line[:content_preview_chars].rstrip()
    out['content'] = f'{snippet}… ({total} chars)'
    return out


def print_tool(name: str, args: dict[str, Any]) -> None:
    ctx = get_context()
    d: AppDefaults = ctx.defaults
    shown = preview_tool_args_for_terminal(
        name,
        args,
        content_preview_chars=d.limits.tool_call_content_preview_chars,
    )
    args_str = ', '.join(f'{k}={repr(v)}' for k, v in shown.items())
    if ctx.console:
        ctx.console.print(f'  [agent.tool]⚙  {name}({args_str})[/]')
    else:
        print(f'  [Tool] -> {name}({args_str})')


def print_error(msg: str) -> None:
    ctx = get_context()
    if ctx.console:
        ctx.console.print(f'[agent.error]✗  {msg}[/]')
    else:
        print(f'ERROR: {msg}')


def print_info(msg: str) -> None:
    ctx = get_context()
    if ctx.console:
        ctx.console.print(f'[agent.info]{msg}[/]')
    else:
        print(msg)


def print_thinking(content: str) -> None:
    """
    Render pre-sanitized model reasoning to the terminal.

    Content is expected to have been processed by ``sanitize_assistant_text``
    before being passed here; this function only handles layout and display.
    """
    if not (content or '').strip():
        return
    t = get_context().defaults.ui.panel_reasoning_title
    _render_markdown_panel(
        content,
        title_rich_markup=f'[agent.info]{t}[/]',
        border_style='dim',
        plain_header=f'\n[{t}]\n',
    )
