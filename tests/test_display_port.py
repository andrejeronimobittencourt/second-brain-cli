"""Tests for brain.repl.display_port."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from brain.repl.display import ReplDisplaySink
from brain.repl.display_port import (
    PlainTtyDisplayPort,
    ReplDisplayPort,
    RichConsoleDisplayPort,
    active_display_port,
    set_display_port,
)


def test_plain_tty_streams_answer(capsys):
    port = PlainTtyDisplayPort()
    set_display_port(port)
    port.begin_answer(plain=True)
    port.write_answer_delta('Hello')
    port.end_answer()
    captured = capsys.readouterr().out
    assert 'Hello' in captured
    assert port.last_answer_streamed_to_tty


def test_repl_port_commits_thinking(ctx):
    sink = ReplDisplaySink(invalidate=MagicMock())
    port = ReplDisplayPort(sink=sink)
    set_display_port(port)
    port.commit_thinking('raw', final_content='Filtered reasoning.')
    assert port.has_committed_thinking()
    assert sink.has_committed_thinking()


def test_rich_console_streams_answer(ctx, monkeypatch):
    from brain.ui import render as ui_mod

    monkeypatch.setattr(ui_mod, '_panel_outer_width', lambda: 80)
    console = Console(file=StringIO(), width=120)
    port = RichConsoleDisplayPort(console)
    set_display_port(port)
    port.begin_answer()
    port.write_answer_delta('Hi')
    port.end_answer()
    assert port.last_answer_streamed_to_tty


def test_active_display_port_never_none(ctx):
    assert active_display_port() is not None


def test_active_display_port_prefers_repl_sink(ctx):
    """REPL transcript must own streaming even when bootstrap left RichConsole active."""
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    ctx.repl_display = sink
    ctx.display = RichConsoleDisplayPort(
        Console(file=StringIO(), width=120),
    )
    port = active_display_port()
    assert isinstance(port, ReplDisplayPort)
    assert port.sink is sink
