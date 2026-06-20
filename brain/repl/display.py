"""Prompt-toolkit output pane for the interactive REPL (Rich panels without cursor escapes)."""

from __future__ import annotations

import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from brain.core.context import get_context

ANSI = None  # type: ignore[misc, assignment]

try:
    from prompt_toolkit.formatted_text import ANSI

    PROMPT_TOOLKIT_ANSI = True
except ImportError:
    PROMPT_TOOLKIT_ANSI = False


@dataclass
class _SpinnerState:
    message: str
    spinner_name: str
    frame: int = 0


def _wrap_prompt_paragraph(paragraph: str, width: int) -> list[str]:
    """Hard-wrap one paragraph for the REPL transcript."""
    if not paragraph.strip():
        return ['']
    wrapped = textwrap.wrap(
        paragraph,
        width=max(8, width),
        break_long_words=True,
        break_on_hyphens=False,
    )
    return wrapped if wrapped else ['']


class ReplDisplaySink:
    """
    Session transcript for the REPL output pane.

    Completed turns accumulate in ``_history`` for the whole session. The active
    turn (user prompt, streaming panel, tool lines) layers on top until
    :meth:`commit_turn`.
    """

    def __init__(self, invalidate: Callable[[], None]) -> None:
        self._invalidate = invalidate
        self._lock = threading.Lock()
        self._history: list[str] = []
        self._turn_blocks: list[str] = []
        self._streaming = False
        self._stream_text = ''
        self._stream_markdown = False
        self._live_ansi = ''
        self._thinking_streaming = False
        self._thinking_text = ''
        self._thinking_raw = ''
        self._thinking_live_ansi = ''
        self._thinking_committed = False
        self._thinking_block_index: int | None = None
        self._spinner: _SpinnerState | None = None

    def has_transcript(self) -> bool:
        with self._lock:
            return bool(
                self._history
                or self._turn_blocks
                or self._live_ansi
                or self._thinking_live_ansi
                or self._streaming
                or self._thinking_streaming
                or self._spinner is not None
            )

    def has_turn_content(self) -> bool:
        with self._lock:
            return bool(
                self._turn_blocks
                or self._live_ansi
                or self._thinking_live_ansi
                or self._streaming
                or self._thinking_streaming
            )

    def has_spinner(self) -> bool:
        with self._lock:
            return self._spinner is not None

    def has_committed_thinking(self) -> bool:
        with self._lock:
            return self._thinking_committed

    def clear_turn(self) -> None:
        """Reset in-flight turn state (history is kept)."""
        with self._lock:
            self._turn_blocks = []
            self._streaming = False
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
            self._thinking_streaming = False
            self._thinking_text = ''
            self._thinking_raw = ''
            self._thinking_live_ansi = ''
            self._thinking_committed = False
            self._thinking_block_index = None
            self._spinner = None
        self._invalidate()

    def append_user_prompt(self, text: str) -> None:
        """Record the user's submitted line(s) in the active turn."""
        from brain.ui.render import render_markup_to_ansi, repl_transcript_inner_width

        ctx = get_context()
        label = ctx.defaults.repl.prompt_label
        inner = repl_transcript_inner_width()
        prefix = f'{label}: '
        continuation = '    '
        first_width = max(8, inner - len(prefix))
        cont_width = max(8, inner - len(continuation))

        markup_lines: list[str] = []
        seen_content = False
        for paragraph in text.split('\n'):
            wrap_w = first_width if not seen_content else cont_width
            for line in _wrap_prompt_paragraph(paragraph, wrap_w):
                if not seen_content:
                    markup_lines.append(f'[bold]{prefix}[/]{line}')
                    seen_content = True
                else:
                    markup_lines.append(f'{continuation}{line}')

        if not markup_lines:
            markup_lines.append(f'[bold]{prefix}[/]')

        markup = '\n'.join(markup_lines)
        ansi = render_markup_to_ansi(markup)
        if not ansi:
            plain = '\n'.join(
                line.replace('[bold]', '').replace('[/]', '')
                for line in markup_lines
            )
            ansi = f'{plain}\n'
        line = ansi if ansi.endswith('\n') else f'{ansi}\n'
        with self._lock:
            self._turn_blocks.append(line)
        self._invalidate()

    def append_session_rich(self, markup: str) -> None:
        """Append a Rich-markup block to permanent session history."""
        from brain.ui.render import render_markup_to_ansi

        ansi = render_markup_to_ansi(markup)
        if not ansi:
            plain = markup.replace('[agent.info]', '').replace('[agent.error]', '')
            plain = plain.replace('[bold]', '').replace('[/]', '')
            ansi = f'{plain}\n' if plain.strip() else ''
        if not ansi:
            return
        line = ansi if ansi.endswith('\n') else f'{ansi}\n'
        with self._lock:
            self._history.append(line)
        self._invalidate()

    def append_session_raw(self, text: str) -> None:
        """Append raw text to permanent session history."""
        if not text:
            return
        line = text if text.endswith('\n') else f'{text}\n'
        with self._lock:
            self._history.append(line)
        self._invalidate()

    def start_spinner(self, message: str, spinner_name: str) -> None:
        with self._lock:
            self._spinner = _SpinnerState(message=message, spinner_name=spinner_name)
        self._invalidate()

    def stop_spinner(self) -> None:
        with self._lock:
            self._spinner = None
        self._invalidate()

    def tick_spinner(self) -> None:
        with self._lock:
            if self._spinner is not None:
                self._spinner.frame += 1

    def append_rich(self, markup: str) -> None:
        """Append one Rich-markup line/block to the active turn."""
        from brain.ui.render import render_markup_to_ansi

        ansi = render_markup_to_ansi(markup)
        if not ansi:
            plain = markup.replace('[agent.info]', '').replace('[agent.error]', '')
            plain = plain.replace('[bold]', '').replace('[/]', '')
            ansi = f'{plain}\n' if plain.strip() else ''
        if not ansi:
            return
        line = ansi if ansi.endswith('\n') else f'{ansi}\n'
        with self._lock:
            self._turn_blocks.append(line)
        self._invalidate()

    def append_info_panel(self, content: str) -> None:
        """Append a dim status panel to the active turn."""
        from brain.ui.render import render_info_panel_to_ansi

        ansi = render_info_panel_to_ansi(content)
        if not ansi:
            return
        line = ansi if ansi.endswith('\n') else f'{ansi}\n'
        with self._lock:
            self._turn_blocks.append(line)
        self._invalidate()

    def append_raw(self, text: str) -> None:
        """Append raw text (for example from ``print()``) to the active turn."""
        if not text:
            return
        line = text if text.endswith('\n') else f'{text}\n'
        with self._lock:
            self._turn_blocks.append(line)
        self._invalidate()

    def has_live_answer(self) -> bool:
        """Return True while a streaming answer panel is visible but not committed."""
        with self._lock:
            return bool(self._live_ansi)

    def begin_answer_stream(self) -> None:
        with self._lock:
            self._streaming = True
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
        self._invalidate()

    def cancel_answer_stream(self) -> None:
        """Drop an in-flight answer stream without committing a panel."""
        with self._lock:
            self._streaming = False
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
        self._invalidate()

    def update_answer_stream(self, content: str, *, markdown: bool = False) -> None:
        from brain.ui.render import render_answer_panel_to_ansi

        if not content.strip():
            return
        ansi = render_answer_panel_to_ansi(content, markdown=markdown)
        if not ansi:
            from brain.ui.render import plain_answer_block

            ansi = plain_answer_block(content)
        with self._lock:
            self._streaming = True
            self._stream_text = content
            self._stream_markdown = markdown
            self._live_ansi = ansi
        self._invalidate()

    def end_answer_stream(self, final_content: str | None = None) -> None:
        from brain.ui.render import plain_answer_block, render_answer_panel_to_ansi

        with self._lock:
            text = final_content if final_content is not None else self._stream_text
            if not text.strip():
                self._streaming = False
                self._stream_text = ''
                self._stream_markdown = False
                self._live_ansi = ''
                self._invalidate()
                return
            markdown = False
            ansi = render_answer_panel_to_ansi(text, markdown=markdown)
            block = ansi if ansi else plain_answer_block(text)
            if block:
                self._turn_blocks.append(block if block.endswith('\n') else f'{block}\n')
            self._streaming = False
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
        self._invalidate()

    def has_live_thinking(self) -> bool:
        with self._lock:
            return bool(self._thinking_live_ansi)

    def has_thinking_content(self) -> bool:
        with self._lock:
            return bool(self._thinking_text.strip() or self._thinking_live_ansi)

    def pending_answer_text(self) -> str:
        with self._lock:
            return self._stream_text

    def pending_thinking_text(self) -> str:
        with self._lock:
            return self._thinking_text

    def pending_thinking_raw(self) -> str:
        with self._lock:
            return self._thinking_raw

    def _filtered_thinking_for_display(self, *, final_content: str | None = None) -> str:
        """Return filtered display text; never raw scaffold."""
        from brain.core.sanitize import extract_display_thinking, sanitize_assistant_text

        if final_content is not None and final_content.strip():
            extracted = extract_display_thinking(final_content.strip(), streaming=False)
            if extracted:
                return sanitize_assistant_text(extracted, get_context())
            return ''
        raw = self._thinking_raw.strip()
        if raw:
            extracted = extract_display_thinking(raw, streaming=False)
            if extracted:
                return sanitize_assistant_text(extracted, get_context())
        if self._thinking_text.strip():
            extracted = extract_display_thinking(self._thinking_text, streaming=False)
            if extracted:
                return sanitize_assistant_text(extracted, get_context())
        return ''

    def _clear_thinking_live(self) -> None:
        self._thinking_streaming = False
        self._thinking_text = ''
        self._thinking_raw = ''
        self._thinking_live_ansi = ''

    def revert_thinking_commit(self) -> None:
        """Remove a committed reasoning panel from the active turn (tool rounds)."""
        with self._lock:
            self._clear_thinking_live()
            if not self._thinking_committed:
                return
            idx = self._thinking_block_index
            if idx is not None and 0 <= idx < len(self._turn_blocks):
                self._turn_blocks.pop(idx)
            self._thinking_committed = False
            self._thinking_block_index = None
        self._invalidate()

    def begin_thinking_stream(self) -> None:
        with self._lock:
            self._thinking_streaming = True
            self._thinking_text = ''
            self._thinking_raw = ''
            self._thinking_live_ansi = ''
        self._invalidate()

    def cancel_thinking_stream(self) -> None:
        with self._lock:
            self._clear_thinking_live()
        self._invalidate()

    def update_thinking_stream(self, content: str, *, raw: str = '') -> None:
        from brain.ui.render import render_thinking_panel_to_ansi

        if not content.strip():
            return
        ansi = render_thinking_panel_to_ansi(content, markdown=False)
        with self._lock:
            self._thinking_streaming = True
            self._thinking_text = content
            if raw:
                self._thinking_raw = raw
            self._thinking_live_ansi = ansi
        self._invalidate()

    def end_thinking_stream(self, final_content: str | None = None) -> None:
        from brain.ui.render import render_thinking_panel_to_ansi

        with self._lock:
            if self._thinking_committed and final_content is None:
                self._clear_thinking_live()
                self._invalidate()
                return
            text = self._filtered_thinking_for_display(final_content=final_content)
            if not text.strip():
                self._clear_thinking_live()
                self._invalidate()
                return
            ansi = render_thinking_panel_to_ansi(text, markdown=False)
            if not ansi:
                self._clear_thinking_live()
                self._invalidate()
                return
            block = ansi if ansi.endswith('\n') else f'{ansi}\n'
            if self._thinking_committed and self._thinking_block_index is not None:
                idx = self._thinking_block_index
                if 0 <= idx < len(self._turn_blocks):
                    self._turn_blocks[idx] = block
                else:
                    self._turn_blocks.append(block)
                    self._thinking_block_index = len(self._turn_blocks) - 1
            else:
                self._turn_blocks.append(block)
                self._thinking_block_index = len(self._turn_blocks) - 1
            self._thinking_committed = True
            self._clear_thinking_live()
        self._invalidate()

    def render_thinking(self, content: str) -> None:
        from brain.ui.render import render_thinking_panel_to_ansi

        text = self._filtered_thinking_for_display(final_content=content)
        if not text.strip():
            return
        ansi = render_thinking_panel_to_ansi(text, markdown=False)
        if not ansi:
            return
        block = ansi if ansi.endswith('\n') else f'{ansi}\n'
        with self._lock:
            if self._thinking_committed and self._thinking_block_index is not None:
                idx = self._thinking_block_index
                if 0 <= idx < len(self._turn_blocks):
                    self._turn_blocks[idx] = block
                else:
                    self._turn_blocks.append(block)
                    self._thinking_block_index = len(self._turn_blocks) - 1
            else:
                self._turn_blocks.append(block)
                self._thinking_block_index = len(self._turn_blocks) - 1
            self._thinking_committed = True
        self._invalidate()

    def render_agent(self, content: str) -> None:
        from brain.ui.render import render_answer_panel_to_ansi

        ansi = render_answer_panel_to_ansi(content, markdown=True)
        if not ansi:
            return
        with self._lock:
            self._turn_blocks.append(ansi if ansi.endswith('\n') else f'{ansi}\n')
        self._invalidate()

    def clear_session(self) -> None:
        """Clear all session transcript blocks (for ``/clear``)."""
        with self._lock:
            self._history = []
            self._turn_blocks = []
            self._streaming = False
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
            self._thinking_streaming = False
            self._thinking_text = ''
            self._thinking_raw = ''
            self._thinking_live_ansi = ''
            self._thinking_committed = False
            self._thinking_block_index = None
            self._spinner = None
        self._invalidate()

    def commit_turn(self) -> None:
        """Append the finished turn to session history and clear in-flight state."""
        with self._lock:
            if self._stream_text.strip() and (self._streaming or self._live_ansi):
                from brain.ui.render import render_answer_panel_to_ansi

                ansi = render_answer_panel_to_ansi(
                    self._stream_text,
                    markdown=False,
                )
                if ansi:
                    block = ansi if ansi.endswith('\n') else f'{ansi}\n'
                    self._turn_blocks.append(block)
            self._streaming = False
            self._stream_text = ''
            self._stream_markdown = False
            self._live_ansi = ''
            self._thinking_streaming = False
            self._thinking_text = ''
            self._thinking_raw = ''
            self._thinking_live_ansi = ''
            self._thinking_committed = False
            self._thinking_block_index = None
            self._history.extend(list(self._turn_blocks))
            self._turn_blocks = []
            self._spinner = None
        self._invalidate()

    def _compose_transcript_ansi(self) -> str:
        from brain.ui.render import (
            plain_answer_block,
            plain_spinner_line,
            plain_thinking_block,
            render_spinner_to_ansi,
        )

        with self._lock:
            parts = list(self._history)
            parts.extend(self._turn_blocks)
            if self._thinking_live_ansi:
                live = self._thinking_live_ansi
                parts.append(live if live.endswith('\n') else f'{live}\n')
            elif self._thinking_streaming and self._thinking_text.strip():
                parts.append(plain_thinking_block(self._thinking_text))
            if self._live_ansi:
                live = self._live_ansi
                parts.append(live if live.endswith('\n') else f'{live}\n')
            elif self._streaming and self._stream_text.strip():
                parts.append(plain_answer_block(self._stream_text))
            if self._spinner is not None:
                spinner = render_spinner_to_ansi(
                    self._spinner.message,
                    self._spinner.frame,
                )
                if not spinner:
                    spinner = plain_spinner_line(
                        self._spinner.message,
                        self._spinner.frame,
                    )
                parts.append(spinner if spinner.endswith('\n') else f'{spinner}\n')
            return ''.join(parts)

    def transcript_line_count(self) -> int:
        """Return the number of terminal rows in the session transcript."""
        return len(self._transcript_lines())

    def _transcript_lines(self) -> list[str]:
        text = self._compose_transcript_ansi()
        if not text:
            return []
        return text.splitlines()

    def visible_transcript_ansi(self, scroll: int, viewport: int) -> tuple[str, int]:
        """
        Return a slice of the transcript for one viewport.

        Returns ``(ansi_text, total_line_count)``. Scroll hints use one row each
        when there is content above or below the visible slice.
        """
        lines = self._transcript_lines()
        total = len(lines)
        if viewport < 1:
            viewport = 1
        if total == 0:
            return '', 0

        if total <= viewport:
            return '\n'.join(lines), total

        max_scroll = max(0, total - max(1, viewport - 1))
        offset = max(0, min(scroll, max_scroll))
        remaining = total - offset

        content_slots = viewport
        if offset > 0:
            content_slots -= 1
        if remaining > content_slots:
            content_slots -= 1
        content_slots = max(1, content_slots)

        visible = lines[offset:offset + content_slots]
        output: list[str] = []
        if offset > 0:
            output.append(f'↑ {offset} more ─────────────────')
        output.extend(visible)
        below = max(0, total - offset - len(visible))
        if below > 0:
            output.append(f'↓ {below} more ─────────────────')
        return '\n'.join(output[:viewport]), total

    def visible_transcript_plain(self, scroll: int, viewport: int) -> tuple[str, int]:
        """Return a plain-text viewport slice (no ANSI) for the transcript buffer."""
        from brain.ui.render import _plain_from_ansi

        ansi, total = self.visible_transcript_ansi(scroll, viewport)
        if not ansi:
            return '', total
        plain_lines = [_plain_from_ansi(line) for line in ansi.splitlines()]
        return '\n'.join(plain_lines), total

    def _to_formatted(self, combined: str) -> Any:
        if not combined:
            return ''
        if ANSI is not None:
            try:
                formatted = ANSI(combined)
                if formatted:
                    return formatted
            except Exception:
                pass
        from brain.ui.render import _plain_from_ansi

        plain = _plain_from_ansi(combined)
        return plain if plain else combined

    def get_transcript_formatted_text(self) -> Any:
        """Return the full session transcript from the top of the output pane."""
        return self._to_formatted(self._compose_transcript_ansi())

    def get_viewport_formatted_text(self, scroll: int, viewport: int) -> Any:
        """Return one visible slice of the transcript for the scrollable viewport."""
        text, _total = self.visible_transcript_ansi(scroll, viewport)
        return self._to_formatted(text)
