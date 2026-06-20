"""Interactive REPL line editor (prompt_toolkit) with history and status bar."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from brain.core.context import get_context
from brain.repl.display import ReplDisplaySink, scroll_hint_line
from brain.repl.display_port import (
    NullDisplayPort,
    PlainTtyDisplayPort,
    ReplDisplayPort,
    RichConsoleDisplayPort,
    set_display_port,
)

logger = logging.getLogger(__name__)

PromptSession = None  # type: ignore[misc, assignment]
FileHistory = None  # type: ignore[misc, assignment]
KeyBindings = None  # type: ignore[misc, assignment]
KeyPressEvent = None  # type: ignore[misc, assignment]
merge_key_bindings = None  # type: ignore[misc, assignment]
load_key_bindings = None  # type: ignore[misc, assignment]
patch_stdout = None  # type: ignore[misc, assignment]
patch_stderr = None  # type: ignore[misc, assignment]
Application = None  # type: ignore[misc, assignment]
Buffer = None  # type: ignore[misc, assignment]
Layout = None  # type: ignore[misc, assignment]
Window = None  # type: ignore[misc, assignment]
BufferControl = None  # type: ignore[misc, assignment]
FormattedTextControl = None  # type: ignore[misc, assignment]
ConditionalContainer = None  # type: ignore[misc, assignment]
HSplit = None  # type: ignore[misc, assignment]
Style = None  # type: ignore[misc, assignment]
FormattedText = None  # type: ignore[misc, assignment]
get_app = None  # type: ignore[misc, assignment]
Dimension = None  # type: ignore[misc, assignment]
Condition = None  # type: ignore[misc, assignment]

# Set while the prompt_toolkit Application is running (worker-safe repaint).
_APP_HOLDER: dict[str, Any] = {'app': None}

try:
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style

    try:
        from prompt_toolkit.patch_stdout import patch_stderr
    except ImportError:
        patch_stderr = None

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    FormattedText = None  # type: ignore[misc, assignment]


class ReplExit(Exception):
    """Raised by slash handlers to leave the REPL loop."""


@dataclass
class ReplStatus:
    """Live session context shown in the bottom toolbar."""

    vault_path: Path
    chat_model: str
    vision_model: str
    session_path: Path


def _toolbar_metadata(status: ReplStatus) -> tuple[str, str, str, str]:
    vault = status.vault_path.name or status.vault_path.as_posix()
    session = status.session_path.name or status.session_path.as_posix()
    return vault, status.chat_model, status.vision_model, session


def format_toolbar(status: ReplStatus) -> str:
    """Format the bottom status line (plain text for tests and toolbar)."""
    vault, chat_model, vision_model, session = _toolbar_metadata(status)
    return (
        f' vault:{vault} | model:{chat_model} | '
        f'vision:{vision_model} | session:{session} '
    )


def refresh_repl_toolbar() -> None:
    """Repaint the footer (safe from worker threads)."""
    refresh = get_context().repl_ui_refresh
    if refresh is not None:
        refresh()
    else:
        _invalidate_ui()


def input_history_path(vault_path: Path) -> Path:
    """Return the vault-local path used for REPL input history."""
    repl = get_context().defaults.repl
    return vault_path / repl.input_history_filename


def _active_app() -> Any | None:
    """Return the running REPL Application (works from worker threads)."""
    return _APP_HOLDER.get('app')


def _schedule_ui_callback(callback: Callable[[], None]) -> None:
    """Run *callback* on the prompt_toolkit UI thread from a worker thread."""
    app = _active_app()
    if app is None:
        return
    try:
        # Do not use loop.is_running() from a worker thread — it is False there
        # even while the Application is active, which caused silent no-ops.
        if getattr(app, '_is_running', False) and app.loop is not None:
            app.loop.call_soon_threadsafe(callback)
            return
        callback()
    except Exception:
        logger.debug('Failed to schedule UI callback', exc_info=True)


def _safe_app_exit(app: Any | None = None) -> None:
    """Request prompt_toolkit shutdown; ignore duplicate ``exit()`` calls."""
    target = app if app is not None else _active_app()
    if target is None:
        return
    try:
        target.exit()
    except Exception:
        logger.debug('Application exit skipped (already exiting)', exc_info=True)


def _request_app_exit() -> None:
    """Leave the prompt_toolkit Application from a worker thread."""
    _schedule_ui_callback(lambda: _safe_app_exit())


def _invalidate_ui() -> None:
    """Repaint the prompt_toolkit Application (safe from the agent worker thread)."""
    app = _active_app()
    if app is None:
        return
    try:
        if not getattr(app, '_is_running', False):
            return
        loop = getattr(app, 'loop', None)
        ui_thread = getattr(app, '_loop_thread', None)
        if (
            loop is not None
            and ui_thread is not None
            and threading.current_thread() is not ui_thread
        ):
            loop.call_soon_threadsafe(app.invalidate)
        else:
            app.invalidate()
    except Exception:
        logger.debug('REPL invalidate failed', exc_info=True)


def _request_transcript_refresh() -> None:
    """Alias kept for callers that expect a transcript-oriented refresh hook."""
    _invalidate_ui()


def set_repl_awaiting_final_answer(waiting: bool) -> None:
    """
    Mark the REPL as waiting for a full agent turn (not per-spinner/stream).

    Input stays locked until this is cleared after the final answer is committed.
    """
    ctx = get_context()
    ctx.repl_awaiting_final_answer = waiting
    refresh = ctx.repl_ui_refresh
    if refresh is not None:
        refresh()
    else:
        _request_transcript_refresh()


def repl_input_is_locked() -> bool:
    """Return True while the user must not edit or submit input."""
    return get_context().repl_awaiting_final_answer


def _stdin_is_interactive() -> bool:
    """Return True when stdin is a TTY suitable for a line editor."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _repl_key_bindings(
    *,
    multiline: bool,
    on_ctrl_c: Callable[[], None],
    is_busy: Callable[[], bool] | None = None,
    on_transcript_page_up: Callable[[], None] | None = None,
    on_transcript_page_down: Callable[[], None] | None = None,
    on_transcript_line_up: Callable[[], None] | None = None,
    on_transcript_line_down: Callable[[], None] | None = None,
    on_transcript_top: Callable[[], None] | None = None,
    on_transcript_bottom: Callable[[], None] | None = None,
) -> KeyBindings | None:
    """Emacs defaults plus Ctrl+C exit and optional multiline submit/newline keys."""
    if KeyBindings is None or merge_key_bindings is None or load_key_bindings is None:
        return None
    custom = KeyBindings()
    busy_check = is_busy if is_busy is not None else repl_input_is_locked
    not_busy = Condition(lambda: not busy_check())

    @custom.add('c-c')
    def _ctrl_c(event: KeyPressEvent) -> None:
        on_ctrl_c()
        _safe_app_exit(event.app)

    if on_transcript_page_up is not None:
        @custom.add('pageup')
        @custom.add('s-pageup')
        def _transcript_page_up(event: KeyPressEvent) -> None:
            on_transcript_page_up()

    if on_transcript_page_down is not None:
        @custom.add('pagedown')
        @custom.add('s-pagedown')
        def _transcript_page_down(event: KeyPressEvent) -> None:
            on_transcript_page_down()

    if on_transcript_line_up is not None:
        @custom.add('c-up')
        @custom.add('escape', 'up')
        @custom.add('s-up')
        def _transcript_line_up(event: KeyPressEvent) -> None:
            on_transcript_line_up()

    if on_transcript_line_down is not None:
        @custom.add('c-down')
        @custom.add('escape', 'down')
        @custom.add('s-down')
        def _transcript_line_down(event: KeyPressEvent) -> None:
            on_transcript_line_down()

    if on_transcript_top is not None:
        @custom.add('home')
        @custom.add('s-home')
        def _transcript_top(event: KeyPressEvent) -> None:
            on_transcript_top()

    if on_transcript_bottom is not None:
        @custom.add('end')
        @custom.add('s-end')
        def _transcript_bottom(event: KeyPressEvent) -> None:
            on_transcript_bottom()

    if multiline:
        @custom.add('enter', filter=not_busy)
        def _submit(event: KeyPressEvent) -> None:
            event.current_buffer.validate_and_handle()

        @custom.add('escape', 'enter', filter=not_busy)
        @custom.add('c-j', filter=not_busy)
        def _newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text('\n')
    else:
        @custom.add('enter', filter=not_busy)
        def _submit_single(event: KeyPressEvent) -> None:
            event.current_buffer.validate_and_handle()

    return merge_key_bindings([load_key_bindings(), custom])


@contextmanager
def patched_repl_output() -> Iterator[None]:
    """Patch stderr while the Application is running (stdout uses the display sink)."""
    if patch_stderr is not None:
        with patch_stderr():
            yield
    else:
        yield


class _ReplStdout:
    """Forward incidental ``print()`` calls into the REPL output pane."""

    def __init__(self, sink: ReplDisplaySink, original: Any) -> None:
        self._sink = sink
        self._original = original

    def write(self, data: str) -> int:
        if data:
            self._sink.append_raw(data)
        return len(data)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return getattr(self._original, 'encoding', 'utf-8') or 'utf-8'


@contextmanager
def _repl_stdout(sink: ReplDisplaySink) -> Iterator[None]:
    """Temporarily route ``sys.stdout`` through the REPL display sink."""
    original = sys.stdout
    sys.stdout = _ReplStdout(sink, original)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original


def repl_prompt_prefix_text(label: str, line_number: int, wrap_count: int) -> str:
    """Return the input prefix for one wrapped row (``You:`` only on the first row)."""
    indent = ' ' * (len(label) + 2)
    if line_number == 0 and wrap_count == 0:
        return f'{label}: '
    return indent


def _buffer_display_line_count(buffer: Buffer, *, label: str, columns: int) -> int:
    """Estimate wrapped input rows for soft-wrapped multiline buffers."""
    prefix_len = len(label) + 2
    total = 0
    for line in buffer.document.lines:
        usable = max(8, columns - prefix_len)
        length = len(line)
        total += max(1, (length + usable - 1) // usable) if length else 1
    return total


def _needed_input_lines(
    buffer: Buffer,
    *,
    label: str,
    columns: int,
    multiline: bool,
    max_lines: int,
) -> int:
    """Return the exact input window height for the current buffer contents."""
    if not multiline:
        return 1
    doc_lines = max(1, buffer.document.line_count)
    wrap_lines = _buffer_display_line_count(buffer, label=label, columns=columns)
    needed = max(doc_lines, wrap_lines)
    return max(1, min(max_lines, needed))


def classify_repl_submit(text: str) -> tuple[bool, str]:
    """
    Classify one submitted buffer for the accept handler.

    Returns ``(is_empty, normalized_text)``. When not empty, the handler must
    leave ``normalized_text`` in the buffer and return ``False`` so
    prompt_toolkit can ``append_to_history()`` before ``reset()``.
    """
    stripped = text.rstrip()
    if not stripped.strip():
        return True, ''
    return False, stripped


def input_hidden_line_counts(
    *,
    total_lines: int,
    window_height: int,
    vertical_scroll: int,
) -> tuple[int, int]:
    """Return ``(hidden_above, hidden_below)`` wrapped display rows in the input window."""
    if window_height < 1 or total_lines <= window_height:
        return 0, 0
    above = max(0, vertical_scroll)
    below = max(0, total_lines - above - window_height)
    return above, below


def _read_input_scroll_hints(input_window: Any | None) -> tuple[int, int]:
    """Read hidden line counts from the last input-window render pass."""
    if input_window is None:
        return 0, 0
    render_info = getattr(input_window, 'render_info', None)
    if render_info is None:
        return 0, 0
    return input_hidden_line_counts(
        total_lines=render_info.ui_content.line_count,
        window_height=render_info.window_height,
        vertical_scroll=render_info.vertical_scroll,
    )


class ReplInput:
    """
    Line editor for the interactive REPL.

    Uses a full-screen prompt_toolkit ``Application`` with a scrollable transcript,
    pinned input line, activity strip, and styled status toolbar at the bottom.
    """

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path.resolve()
        self._status = ReplStatus(
            vault_path=self._vault_path,
            chat_model='',
            vision_model='',
            session_path=Path('.agent_history.json'),
        )
        self._uses_ptk = PROMPT_TOOLKIT_AVAILABLE and _stdin_is_interactive()
        self._init_error: str | None = None
        self._busy = False
        if not PROMPT_TOOLKIT_AVAILABLE:
            self._init_error = 'prompt_toolkit is not installed'
        elif not _stdin_is_interactive():
            self._init_error = 'stdin is not interactive'

    @property
    def uses_line_editor(self) -> bool:
        """True when prompt_toolkit drives input (arrow keys, history, toolbar)."""
        return self._uses_ptk

    @property
    def fallback_reason(self) -> str | None:
        """Human-readable reason for basic ``input()`` fallback, if any."""
        if self._uses_ptk:
            return None
        return self._init_error

    def _styled_toolbar(self) -> FormattedText | str:
        text = format_toolbar(self._status)
        if FormattedText is None:
            return text
        style = get_context().defaults.repl.toolbar_style
        return FormattedText([('class:bottom-toolbar', text)])

    def run(
        self,
        *,
        status_fn: Callable[[], ReplStatus],
        process_line: Callable[[str], None],
        on_exit: Callable[[], None],
        seed_display: Callable[[ReplDisplaySink], None] | None = None,
        farewell_message: str | None = None,
    ) -> None:
        """
        Run the interactive loop until EOF, Ctrl+C, or :class:`ReplExit`.

        Session output accumulates in the scrollable transcript pane; the input
        line and status toolbar stay pinned at the bottom of the terminal.
        """
        exit_requested = False

        def _wrapped_on_exit() -> None:
            nonlocal exit_requested
            exit_requested = True
            on_exit()

        if not self._uses_ptk or Application is None or Buffer is None:
            self._run_fallback(status_fn, process_line, _wrapped_on_exit)
        else:
            self._run_application(
                status_fn=status_fn,
                process_line=process_line,
                on_exit=_wrapped_on_exit,
                seed_display=seed_display,
            )

        if exit_requested and farewell_message:
            from brain.ui.render import print_terminal_info

            print_terminal_info(farewell_message)

    def _run_application(
        self,
        *,
        status_fn: Callable[[], ReplStatus],
        process_line: Callable[[str], None],
        on_exit: Callable[[], None],
        seed_display: Callable[[ReplDisplaySink], None] | None,
    ) -> None:
        """Run the full-screen prompt_toolkit application."""
        repl_cfg = get_context().defaults.repl
        label = repl_cfg.prompt_label
        history_path = input_history_path(self._vault_path)
        ctx = get_context()
        display = ReplDisplaySink(invalidate=_invalidate_ui)
        ctx.repl_display = display
        set_display_port(ReplDisplayPort(sink=display))
        ctx.repl_ui_refresh = _invalidate_ui

        def _set_input_locked(locked: bool) -> None:
            """Block editing until the full agent turn completes (final answer)."""
            self._busy = locked
            ctx.repl_awaiting_final_answer = locked
            _invalidate_ui()

        def _ctrl_c_exit() -> None:
            on_exit()

        def _accept(buff: Buffer) -> bool:
            if self._busy or repl_input_is_locked():
                return True
            is_empty, normalized = classify_repl_submit(buff.text)
            if is_empty:
                buff.text = ''
                layout_state['input_height'] = 1
                input_window = layout_state.get('input_window')
                if input_window is not None:
                    input_window.height = 1
                return True

            buff.text = normalized
            display.append_user_prompt(normalized)
            _set_input_locked(True)

            def _worker() -> None:
                try:
                    ctx.repl_line_editor = True
                    with patched_repl_output(), _repl_stdout(display):
                        set_display_port(ReplDisplayPort(sink=display))
                        try:
                            process_line(normalized)
                        finally:
                            get_context().display.flush_turn()
                except ReplExit:
                    on_exit()
                    _request_app_exit()
                except KeyboardInterrupt:
                    on_exit()
                    _request_app_exit()
                except Exception:
                    logger.exception('REPL turn failed')
                finally:
                    _schedule_ui_callback(lambda: _set_input_locked(False))

            threading.Thread(target=_worker, daemon=True).start()
            # False → prompt_toolkit append_to_history() then reset().
            return False

        history = FileHistory(str(history_path)) if FileHistory is not None else None
        input_locked = Condition(repl_input_is_locked)
        buffer = Buffer(
            history=history,
            accept_handler=_accept,
            read_only=input_locked,
        )

        layout_state: dict[str, Any] = {
            'input_height': 1,
            'input_above': 0,
            'input_below': 0,
            'transcript_viewport': 1,
            'transcript_scroll': 0,
            'transcript_lines': 0,
            'transcript_follow': True,
            'input_window': None,
        }

        def _input_hint_row_count() -> int:
            return int(layout_state['input_above'] > 0) + int(layout_state['input_below'] > 0)

        def _update_input_scroll_hints() -> bool:
            if not repl_cfg.multiline_input or self._busy:
                above = below = 0
            else:
                above, below = _read_input_scroll_hints(layout_state.get('input_window'))
            changed = (
                layout_state['input_above'] != above
                or layout_state['input_below'] != below
            )
            layout_state['input_above'] = above
            layout_state['input_below'] = below
            return changed

        def _max_transcript_scroll() -> int:
            total = layout_state['transcript_lines']
            viewport = layout_state['transcript_viewport']
            if total <= viewport:
                return 0
            return max(0, total - max(1, viewport - 1))

        def _set_transcript_scroll(offset: int, *, follow: bool) -> None:
            layout_state['transcript_follow'] = follow
            layout_state['transcript_scroll'] = max(
                0,
                min(_max_transcript_scroll(), offset),
            )
            if get_app is not None:
                _invalidate_ui()

        def _scroll_transcript_page(delta: int) -> None:
            page = max(1, layout_state['transcript_viewport'] - 2)
            _set_transcript_scroll(
                layout_state['transcript_scroll'] + delta * page,
                follow=False,
            )

        def _scroll_transcript_line(delta: int) -> None:
            _set_transcript_scroll(
                layout_state['transcript_scroll'] + delta,
                follow=False,
            )

        def _sync_layout_heights() -> None:
            app = _active_app()
            if app is None:
                return
            try:
                term_rows = app.output.get_size().rows
                columns = app.output.get_size().columns
            except Exception:
                term_rows = 24
                columns = 80

            ctx.repl_terminal_columns = columns

            input_h = 1 if self._busy else _needed_input_lines(
                buffer,
                label=label,
                columns=columns,
                multiline=repl_cfg.multiline_input,
                max_lines=repl_cfg.multiline_max_lines,
            )
            toolbar_h = 1
            hint_rows = _input_hint_row_count()
            viewport = max(1, term_rows - input_h - hint_rows - toolbar_h)

            _text, total_lines = display.visible_transcript_ansi(0, viewport)
            layout_state['transcript_viewport'] = viewport
            layout_state['transcript_lines'] = total_lines
            max_scroll = _max_transcript_scroll()
            if layout_state['transcript_follow']:
                layout_state['transcript_scroll'] = max_scroll
            else:
                layout_state['transcript_scroll'] = min(
                    layout_state['transcript_scroll'],
                    max_scroll,
                )
                if layout_state['transcript_scroll'] >= max_scroll:
                    layout_state['transcript_follow'] = True

            changed = False
            if layout_state['input_height'] != input_h:
                layout_state['input_height'] = input_h
                input_window = layout_state.get('input_window')
                if input_window is not None:
                    input_window.height = input_h
                changed = True
            if changed:
                app.invalidate()

        kb = _repl_key_bindings(
            multiline=repl_cfg.multiline_input,
            on_ctrl_c=_ctrl_c_exit,
            is_busy=repl_input_is_locked,
            on_transcript_page_up=lambda: _scroll_transcript_page(-1),
            on_transcript_page_down=lambda: _scroll_transcript_page(1),
            on_transcript_line_up=lambda: _scroll_transcript_line(-1),
            on_transcript_line_down=lambda: _scroll_transcript_line(1),
            on_transcript_top=lambda: _set_transcript_scroll(0, follow=False),
            on_transcript_bottom=lambda: _set_transcript_scroll(
                _max_transcript_scroll(),
                follow=True,
            ),
        )

        if (
            kb is None
            or ConditionalContainer is None
            or FormattedTextControl is None
            or HSplit is None
            or Window is None
            or Dimension is None
            or Condition is None
        ):
            ctx.repl_display = None
            self._run_fallback(status_fn, process_line, on_exit)
            return

        def _bottom_toolbar() -> FormattedText | str:
            self._status = status_fn()
            return self._styled_toolbar()

        def _transcript_text() -> FormattedText | str:
            scroll = layout_state['transcript_scroll']
            viewport = layout_state['transcript_viewport']
            return display.get_viewport_formatted_text(scroll, viewport)

        def _prompt_prefix(line_number: int, wrap_count: int) -> FormattedText | str:
            if repl_input_is_locked() and line_number == 0 and wrap_count == 0:
                message = repl_cfg.waiting_for_answer_message
                if FormattedText is not None:
                    return FormattedText([('class:waiting', message)])
                return message
            prefix = repl_prompt_prefix_text(label, line_number, wrap_count)
            if line_number == 0 and wrap_count == 0 and FormattedText is not None:
                return FormattedText([('class:prompt', prefix)])
            return prefix

        def _input_hint_above_text() -> FormattedText | str:
            count = layout_state['input_above']
            if count <= 0:
                return ''
            return scroll_hint_line(direction='up', count=count)

        def _input_hint_below_text() -> FormattedText | str:
            count = layout_state['input_below']
            if count <= 0:
                return ''
            return scroll_hint_line(direction='down', count=count)

        input_hint_above_filter = Condition(lambda: layout_state['input_above'] > 0)
        input_hint_below_filter = Condition(lambda: layout_state['input_below'] > 0)

        def _make_root() -> HSplit:
            transcript_window = Window(
                FormattedTextControl(_transcript_text),
                wrap_lines=False,
                height=Dimension(weight=1),
            )
            layout_state['transcript_window'] = transcript_window
            input_window = Window(
                BufferControl(buffer=buffer, focusable=True),
                get_line_prefix=_prompt_prefix,
                wrap_lines=repl_cfg.multiline_input,
                height=layout_state['input_height'],
            )
            layout_state['input_window'] = input_window
            input_stack = HSplit([
                ConditionalContainer(
                    Window(
                        FormattedTextControl(_input_hint_above_text),
                        height=1,
                        style='class:input-scroll-hint',
                    ),
                    filter=input_hint_above_filter,
                ),
                input_window,
                ConditionalContainer(
                    Window(
                        FormattedTextControl(_input_hint_below_text),
                        height=1,
                        style='class:input-scroll-hint',
                    ),
                    filter=input_hint_below_filter,
                ),
            ])
            toolbar_window = Window(
                FormattedTextControl(_bottom_toolbar),
                height=1,
                style='class:bottom-toolbar',
            )
            return HSplit([
                transcript_window,
                input_stack,
                toolbar_window,
            ])

        def _resize_input_buffer(_: object) -> None:
            _sync_layout_heights()

        buffer.on_text_changed.add_handler(_resize_input_buffer)

        root = _make_root()

        style = Style.from_dict({
            'bottom-toolbar': repl_cfg.toolbar_style,
            'input-scroll-hint': 'ansibrightblack',
            'prompt': 'bold',
            'waiting': 'bold ansiyellow',
        }) if Style is not None else None

        def _before_render(*_args: object) -> None:
            if display.has_spinner():
                display.tick_spinner()
            hints_changed = _update_input_scroll_hints()
            _sync_layout_heights()
            if hints_changed:
                app = _active_app()
                if app is not None:
                    app.invalidate()

        def _pre_run() -> None:
            if seed_display is not None:
                seed_display(display)
            _invalidate_ui()

        app = Application(
            layout=Layout(root, focused_element=buffer),
            key_bindings=kb,
            full_screen=True,
            style=style,
            refresh_interval=0.05,
            before_render=_before_render,
        )

        try:
            _APP_HOLDER['app'] = app
            app.run(pre_run=_pre_run)
        except KeyboardInterrupt:
            on_exit()
        except ReplExit:
            pass
        finally:
            _APP_HOLDER['app'] = None
            ctx.repl_display = None
            ctx.display = NullDisplayPort()
            ctx.repl_ui_refresh = None
            ctx.repl_awaiting_final_answer = False
            ctx.repl_terminal_columns = 0

    def _run_fallback(
        self,
        status_fn: Callable[[], ReplStatus],
        process_line: Callable[[str], None],
        on_exit: Callable[[], None],
    ) -> None:
        """Run basic ``input()`` when prompt_toolkit is unavailable."""
        ctx = get_context()
        if ctx.console is not None:
            set_display_port(RichConsoleDisplayPort(ctx.console))
        else:
            set_display_port(PlainTtyDisplayPort())
        label = get_context().defaults.repl.prompt_label
        while True:
            self._status = status_fn()
            try:
                text = input(f'{label}: ')
            except EOFError:
                on_exit()
                break
            except KeyboardInterrupt:
                on_exit()
                break
            stripped = text.strip()
            if not stripped:
                continue
            try:
                process_line(stripped)
            except ReplExit:
                on_exit()
                break

    def prompt(self, *, status: ReplStatus) -> str | None:
        """Legacy single-shot prompt (fallback path only)."""
        self._status = status
        label = get_context().defaults.repl.prompt_label
        try:
            text = input(f'{label}: ')
        except EOFError:
            return None
        except KeyboardInterrupt:
            raise
        return text.strip()


def create_repl_input(vault_path: Path) -> ReplInput:
    """Build a REPL input session for *vault_path*."""
    return ReplInput(vault_path)
