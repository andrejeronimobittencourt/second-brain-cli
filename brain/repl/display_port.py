"""Single-owner display layer for REPL streaming panels."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from brain.core.context import get_context
from brain.core.sanitize import extract_display_thinking, sanitize_assistant_text
from brain.core.streaming import merge_stream_content

if TYPE_CHECKING:
    from rich.console import Console

    from brain.repl.display import ReplDisplaySink


class TurnPhase(str, Enum):
    """Lifecycle of one REPL turn display commit."""

    IDLE = 'idle'
    STREAMING = 'streaming'
    COMMITTED = 'committed'


class FinalizePolicy(str, Enum):
    """When to apply markdown formatting on answer finalize."""

    IMMEDIATE = 'immediate'
    DEFER_MARKDOWN = 'defer_markdown'


@dataclass
class ThinkingStreamState:
    """Filtered reasoning stream buffers."""

    active: bool = False
    raw: str = ''
    display: str = ''
    displayed_prefix: str = ''


@dataclass
class AnswerStreamState:
    """Assistant answer stream buffers."""

    active: bool = False
    buffer: str = ''
    plain: bool = False
    markdown_pending: bool = False


def _sync_context_flags(port: DisplayPort) -> None:
    """Mirror display turn flags onto ``ApplicationContext``."""
    ctx = get_context()
    ctx.last_answer_streamed_to_tty = port.last_answer_streamed_to_tty
    ctx.last_thinking_streamed_to_tty = port.last_thinking_streamed_to_tty
    ctx.thinking_committed_this_turn = port.thinking_committed_this_turn


class DisplayPort(ABC):
    """Abstract display owner for one agent session."""

    turn_phase: TurnPhase = TurnPhase.IDLE
    finalize_policy: FinalizePolicy = FinalizePolicy.DEFER_MARKDOWN
    last_answer_streamed_to_tty: bool = False
    last_thinking_streamed_to_tty: bool = False
    thinking_committed_this_turn: bool = False

    @abstractmethod
    def reset_turn(self) -> None:
        """Clear per-turn streaming flags."""

    @abstractmethod
    def sync_thinking_display(self, raw_thinking: str, *, displayed_prefix: str) -> str:
        """Stream filtered reasoning; return prefix already shown."""

    @abstractmethod
    def commit_thinking(self, raw_thinking: str, *, final_content: str | None = None) -> None:
        """Finalize filtered reasoning once per turn."""

    @abstractmethod
    def cancel_thinking(self) -> None:
        """Drop in-flight reasoning without committing."""

    @abstractmethod
    def thinking_is_active(self) -> bool:
        """Return True while reasoning is streaming."""

    @abstractmethod
    def begin_answer(self, *, plain: bool = False) -> None:
        """Start streaming an assistant reply."""

    @abstractmethod
    def write_answer_delta(self, text: str) -> None:
        """Append one answer chunk."""

    @abstractmethod
    def end_answer(self, *, final_content: str | None = None) -> None:
        """Finish the active answer stream."""

    @abstractmethod
    def cancel_answer(self) -> None:
        """Abort the active answer stream."""

    @abstractmethod
    def answer_is_active(self) -> bool:
        """Return True while an answer stream is in progress."""

    @abstractmethod
    def flush_turn(self) -> None:
        """Idempotent end-of-turn commit into transcript history."""

    def discard_thinking(self) -> None:
        """Drop buffered reasoning without keeping a panel (tool rounds)."""
        self.cancel_thinking()
        self.thinking_committed_this_turn = False
        self.last_thinking_streamed_to_tty = False
        _sync_context_flags(self)

    def has_committed_thinking(self) -> bool:
        """Return True when reasoning was committed this turn."""
        return self.thinking_committed_this_turn

    def reveal_thinking_before_answer(self, raw_thinking: str) -> None:
        """Progressively show buffered reasoning when answer content begins."""

    def print_thinking_panel(self, content: str) -> None:
        """Render a one-shot reasoning panel when nothing was streamed live."""

    def print_agent_panel(self, content: str) -> None:
        """Render a one-shot assistant answer panel."""

    def has_live_answer(self) -> bool:
        """Return True when a live answer panel is open (REPL sink path)."""
        return False


class NullDisplayPort(DisplayPort):
    """No-op display port used before REPL starts."""

    def reset_turn(self) -> None:
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.turn_phase = TurnPhase.IDLE
        _sync_context_flags(self)

    def sync_thinking_display(self, raw_thinking: str, *, displayed_prefix: str) -> str:
        return displayed_prefix

    def commit_thinking(self, raw_thinking: str, *, final_content: str | None = None) -> None:
        return

    def cancel_thinking(self) -> None:
        return

    def thinking_is_active(self) -> bool:
        return False

    def begin_answer(self, *, plain: bool = False) -> None:
        return

    def write_answer_delta(self, text: str) -> None:
        return

    def end_answer(self, *, final_content: str | None = None) -> None:
        return

    def cancel_answer(self) -> None:
        return

    def answer_is_active(self) -> bool:
        return False

    def flush_turn(self) -> None:
        self.turn_phase = TurnPhase.COMMITTED


@dataclass
class ReplDisplayPort(DisplayPort):
    """
    REPL transcript owner: single source of truth for stream buffers.

    Throttles live panel re-renders during deltas (max ~10 Hz).
    """

    sink: ReplDisplaySink
    thinking: ThinkingStreamState = field(default_factory=ThinkingStreamState)
    answer: AnswerStreamState = field(default_factory=AnswerStreamState)
    _min_render_interval_s: float = 0.05
    _last_thinking_render: float = 0.0
    _last_answer_render: float = 0.0

    def reset_turn(self) -> None:
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.turn_phase = TurnPhase.IDLE
        self.thinking = ThinkingStreamState()
        self.answer = AnswerStreamState()
        _sync_context_flags(self)

    def _filtered_display(self, raw_thinking: str, *, streaming: bool = False) -> str:
        extracted = extract_display_thinking(raw_thinking, streaming=streaming)
        if not extracted:
            return ''
        return sanitize_assistant_text(extracted, get_context())

    def sync_thinking_display(self, raw_thinking: str, *, displayed_prefix: str) -> str:
        self.thinking.raw = raw_thinking
        display = self._filtered_display(raw_thinking, streaming=True)
        if not display:
            return displayed_prefix
        if display == displayed_prefix:
            return displayed_prefix
        prefix = displayed_prefix
        if prefix and not display.startswith(prefix):
            self.cancel_thinking()
            prefix = ''
        self.thinking.display = display
        self.thinking.displayed_prefix = display
        if not self.thinking.active:
            self.thinking.active = True
            self.sink.begin_thinking_stream()
            self.turn_phase = TurnPhase.STREAMING
        now = time.monotonic()
        if now - self._last_thinking_render >= self._min_render_interval_s:
            self._push_thinking_live(display, raw=raw_thinking)
            self._last_thinking_render = now
        return display

    def _push_thinking_live(self, display: str, *, raw: str) -> None:
        self.sink.update_thinking_stream(display, raw=raw)
        self.last_thinking_streamed_to_tty = True
        _sync_context_flags(self)

    def commit_thinking(self, raw_thinking: str, *, final_content: str | None = None) -> None:
        if self.thinking_committed_this_turn and final_content is None:
            self.cancel_thinking()
            return
        if (
            not self.thinking_committed_this_turn
            and self.thinking.active
            and self.thinking.display.strip()
        ):
            self.sink.update_thinking_stream(
                self.thinking.display,
                raw=raw_thinking or self.thinking.raw,
            )
        if final_content is not None:
            text = final_content.strip()
        else:
            text = self._filtered_display(raw_thinking or self.thinking.raw, streaming=False)
        if not text:
            self.cancel_thinking()
            return
        self.sink.end_thinking_stream(final_content=text)
        self.thinking_committed_this_turn = True
        self.thinking.active = False
        self.thinking.raw = ''
        self.thinking.display = ''
        self.thinking.displayed_prefix = ''
        _sync_context_flags(self)

    def cancel_thinking(self) -> None:
        self.sink.cancel_thinking_stream()
        self.thinking = ThinkingStreamState()

    def discard_thinking(self) -> None:
        self.sink.revert_thinking_commit()
        self.cancel_thinking()
        self.thinking_committed_this_turn = False
        self.last_thinking_streamed_to_tty = False
        _sync_context_flags(self)

    def has_committed_thinking(self) -> bool:
        return self.thinking_committed_this_turn or self.sink.has_committed_thinking()

    def thinking_is_active(self) -> bool:
        return self.thinking.active

    def reveal_thinking_before_answer(self, raw_thinking: str) -> None:
        if not raw_thinking.strip() or self.thinking_committed_this_turn:
            return
        if self.sink.has_committed_thinking():
            return
        text = self._filtered_display(raw_thinking, streaming=False)
        if not text:
            return
        if not self.thinking.active:
            self.thinking.active = True
            self.sink.begin_thinking_stream()
            self.turn_phase = TurnPhase.STREAMING
        steps = max(8, min(32, len(text) // 4))
        step = max(1, (len(text) + steps - 1) // steps)
        end = step
        while end <= len(text):
            partial = text[:end]
            self._push_thinking_live(partial, raw=raw_thinking)
            self.thinking.display = partial
            self.thinking.displayed_prefix = partial
            end += step
        if self.thinking.display != text:
            self._push_thinking_live(text, raw=raw_thinking)
            self.thinking.display = text
            self.thinking.displayed_prefix = text

    def print_thinking_panel(self, content: str) -> None:
        if not content.strip():
            return
        self.sink.render_thinking(content)
        self.thinking_committed_this_turn = True
        self.last_thinking_streamed_to_tty = True
        _sync_context_flags(self)

    def print_agent_panel(self, content: str) -> None:
        if self.answer.active or self.sink.has_live_answer():
            self.end_answer(final_content=content if content else None)
            return
        self.sink.render_agent(content)

    def has_live_answer(self) -> bool:
        return self.sink.has_live_answer()

    def begin_answer(self, *, plain: bool = False) -> None:
        if self.answer.active:
            self.cancel_answer()
        self.answer = AnswerStreamState(active=True, plain=plain)
        self.sink.begin_answer_stream()
        self.turn_phase = TurnPhase.STREAMING

    def write_answer_delta(self, text: str) -> None:
        if not self.answer.active or not text:
            return
        self.answer.buffer = merge_stream_content(self.answer.buffer, text)
        if not self.answer.buffer.strip():
            return
        now = time.monotonic()
        if now - self._last_answer_render >= self._min_render_interval_s:
            self.sink.update_answer_stream(self.answer.buffer, markdown=False)
            self._last_answer_render = now
            self.last_answer_streamed_to_tty = True
            _sync_context_flags(self)

    def end_answer(self, *, final_content: str | None = None) -> None:
        if self.answer.active and self.answer.buffer.strip():
            self.sink.update_answer_stream(self.answer.buffer, markdown=False)
        text = (
            final_content if final_content is not None else self.answer.buffer
        ).strip()
        if not text and not self.sink.has_live_answer():
            self.cancel_answer()
            return
        if text or self.sink.has_live_answer():
            self.sink.end_answer_stream(
                final_content if final_content is not None else (text or None),
            )
            self.last_answer_streamed_to_tty = True
            _sync_context_flags(self)
        self.answer = AnswerStreamState()

    def cancel_answer(self) -> None:
        self.sink.cancel_answer_stream()
        self.answer = AnswerStreamState()

    def answer_is_active(self) -> bool:
        return self.answer.active

    def flush_turn(self) -> None:
        if self.turn_phase == TurnPhase.COMMITTED:
            return
        self.sink.commit_turn()
        self.turn_phase = TurnPhase.COMMITTED


class RichConsoleDisplayPort(DisplayPort):
    """Rich ``Live`` panel streaming when no REPL sink is active."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self.turn_phase = TurnPhase.IDLE
        self.finalize_policy = FinalizePolicy.IMMEDIATE
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.thinking = ThinkingStreamState()
        self.answer = AnswerStreamState()
        self._answer_live: Any = None

    def reset_turn(self) -> None:
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.turn_phase = TurnPhase.IDLE
        self.thinking = ThinkingStreamState()
        self.answer = AnswerStreamState()
        _sync_context_flags(self)

    def _filtered_display(self, raw_thinking: str, *, streaming: bool = False) -> str:
        extracted = extract_display_thinking(raw_thinking, streaming=streaming)
        if not extracted:
            return ''
        return sanitize_assistant_text(extracted, get_context())

    def sync_thinking_display(self, raw_thinking: str, *, displayed_prefix: str) -> str:
        self.thinking.raw = raw_thinking
        display = self._filtered_display(raw_thinking, streaming=True)
        if not display:
            return displayed_prefix
        if display == displayed_prefix:
            return displayed_prefix
        prefix = displayed_prefix
        if prefix and not display.startswith(prefix):
            self.cancel_thinking()
            prefix = ''
        if not self.thinking.active:
            self.thinking.active = True
            self.turn_phase = TurnPhase.STREAMING
        self.thinking.display = display
        self.thinking.displayed_prefix = display
        self.last_thinking_streamed_to_tty = True
        _sync_context_flags(self)
        return display

    def commit_thinking(self, raw_thinking: str, *, final_content: str | None = None) -> None:
        if self.thinking_committed_this_turn and final_content is None:
            self.cancel_thinking()
            return
        if final_content is not None:
            text = final_content.strip()
        else:
            text = self._filtered_display(raw_thinking or self.thinking.raw, streaming=False)
        if not text:
            self.cancel_thinking()
            return
        from brain.ui.render import _render_thinking_panel

        _render_thinking_panel(text)
        self.thinking_committed_this_turn = True
        self.thinking = ThinkingStreamState()
        _sync_context_flags(self)

    def cancel_thinking(self) -> None:
        self.thinking = ThinkingStreamState()

    def thinking_is_active(self) -> bool:
        return self.thinking.active

    def print_thinking_panel(self, content: str) -> None:
        from brain.ui.render import _render_thinking_panel

        if not content.strip():
            return
        _render_thinking_panel(content)
        self.thinking_committed_this_turn = True
        self.last_thinking_streamed_to_tty = True
        _sync_context_flags(self)

    def print_agent_panel(self, content: str) -> None:
        from brain.ui.render import _render_agent_panel

        if self.answer.active:
            self.end_answer(final_content=content if content else None)
            return
        _render_agent_panel(content)

    def begin_answer(self, *, plain: bool = False) -> None:
        from brain.ui.render import RICH_AVAILABLE, Live, _answer_panel

        if get_context().repl_display is not None:
            return
        if self.answer.active:
            self.cancel_answer()
        self.answer = AnswerStreamState(active=True, plain=plain)
        self.turn_phase = TurnPhase.STREAMING
        title = get_context().defaults.ui.panel_answer_title
        if plain or not RICH_AVAILABLE or Live is None:
            sys.stdout.write(f'\n{title}:\n')
            sys.stdout.flush()
            return
        self._console.file = sys.stdout
        self._answer_live = Live(
            _answer_panel('', markdown=False),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._answer_live.start()

    def write_answer_delta(self, text: str) -> None:
        from brain.ui.render import _answer_panel

        if get_context().repl_display is not None:
            return
        if not self.answer.active or not text:
            return
        self.answer.buffer = merge_stream_content(self.answer.buffer, text)
        if not self.answer.buffer.strip():
            return
        if self._answer_live is not None:
            self._answer_live.update(_answer_panel(self.answer.buffer, markdown=False))
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        self.last_answer_streamed_to_tty = True
        _sync_context_flags(self)

    def end_answer(self, *, final_content: str | None = None) -> None:
        from brain.ui.render import _answer_panel

        content = final_content if final_content is not None else self.answer.buffer
        if self._answer_live is not None:
            self._answer_live.update(_answer_panel(content, markdown=False))
            self._answer_live.stop()
            self._answer_live = None
        elif self.answer.active:
            sys.stdout.write('\n')
            sys.stdout.flush()
        self.answer = AnswerStreamState()
        self.last_answer_streamed_to_tty = True
        _sync_context_flags(self)

    def cancel_answer(self) -> None:
        if self._answer_live is not None:
            self._answer_live.stop()
            self._answer_live = None
        self.answer = AnswerStreamState()

    def answer_is_active(self) -> bool:
        return self.answer.active

    def flush_turn(self) -> None:
        self.turn_phase = TurnPhase.COMMITTED


class PlainTtyDisplayPort(DisplayPort):
    """Stdout streaming for ``--print`` mode (no REPL sink)."""

    def __init__(self) -> None:
        self.turn_phase = TurnPhase.IDLE
        self.finalize_policy = FinalizePolicy.IMMEDIATE
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.thinking = ThinkingStreamState()
        self.answer = AnswerStreamState()

    def reset_turn(self) -> None:
        self.last_answer_streamed_to_tty = False
        self.last_thinking_streamed_to_tty = False
        self.thinking_committed_this_turn = False
        self.turn_phase = TurnPhase.IDLE
        self.thinking = ThinkingStreamState()
        self.answer = AnswerStreamState()
        _sync_context_flags(self)

    def sync_thinking_display(self, raw_thinking: str, *, displayed_prefix: str) -> str:
        return displayed_prefix

    def commit_thinking(self, raw_thinking: str, *, final_content: str | None = None) -> None:
        return

    def cancel_thinking(self) -> None:
        return

    def thinking_is_active(self) -> bool:
        return False

    def begin_answer(self, *, plain: bool = False) -> None:
        if self.answer.active:
            self.cancel_answer()
        self.answer = AnswerStreamState(active=True, plain=True)
        title = get_context().defaults.ui.panel_answer_title
        sys.stdout.write(f'\n{title}:\n')
        sys.stdout.flush()
        self.turn_phase = TurnPhase.STREAMING

    def write_answer_delta(self, text: str) -> None:
        if not self.answer.active or not text:
            return
        self.answer.buffer = merge_stream_content(self.answer.buffer, text)
        sys.stdout.write(text)
        sys.stdout.flush()
        self.last_answer_streamed_to_tty = True
        _sync_context_flags(self)

    def end_answer(self, *, final_content: str | None = None) -> None:
        if self.answer.active:
            sys.stdout.write('\n')
            sys.stdout.flush()
        self.answer = AnswerStreamState()
        self.last_answer_streamed_to_tty = True
        _sync_context_flags(self)

    def cancel_answer(self) -> None:
        self.answer = AnswerStreamState()

    def answer_is_active(self) -> bool:
        return self.answer.active

    def flush_turn(self) -> None:
        self.turn_phase = TurnPhase.COMMITTED


def _repl_port_for_active_sink(sink: ReplDisplaySink) -> ReplDisplayPort:
    """Return the REPL port bound to *sink*, installing it when needed."""
    ctx = get_context()
    port = ctx.display
    if isinstance(port, ReplDisplayPort) and port.sink is sink:
        return port
    repl_port = ReplDisplayPort(sink=sink)
    ctx.display = repl_port
    _sync_context_flags(repl_port)
    return repl_port


def active_display_port() -> DisplayPort:
    """Return the session display port (never ``None``)."""
    ctx = get_context()
    sink = ctx.repl_display
    if sink is not None:
        return _repl_port_for_active_sink(sink)
    port = ctx.display
    if port is None:
        return NullDisplayPort()
    return port


def set_display_port(port: DisplayPort) -> None:
    """Install *port* as the session display owner."""
    get_context().display = port
    _sync_context_flags(port)
