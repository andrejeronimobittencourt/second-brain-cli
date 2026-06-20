"""Tests for ``brain.ui.render`` helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from brain.core.defaults import APP_DEFAULTS
from brain.repl.display import ReplDisplaySink
from brain.repl.display_port import ReplDisplayPort
from brain.ui.render import (
    begin_answer_stream,
    cancel_answer_stream,
    end_answer_stream,
    format_tool_call_line,
    preview_tool_args_for_terminal,
    print_thinking,
    write_answer_delta,
)


def _wire_repl_sink(ctx, sink: ReplDisplaySink) -> None:
    ctx.repl_display = sink
    ctx.display = ReplDisplayPort(sink=sink)


def test_preview_truncates_create_note_content():
    long = 'a' * 500
    out = preview_tool_args_for_terminal(
        'create_note',
        {'note_title': 'T', 'folder': 'F', 'content': long},
        content_preview_chars=100,
    )
    assert out['note_title'] == 'T'
    assert out['folder'] == 'F'
    assert out['content'] == '500 chars'
    assert 'aaa' not in out['content']


def test_preview_multiline_content_shows_char_count_only():
    content = 'First line\nSecond line\nThird line'
    out = preview_tool_args_for_terminal(
        'create_note',
        {'note_title': 'T', 'content': content},
        content_preview_chars=80,
    )
    assert out['content'] == f'{len(content)} chars'
    assert 'First line' not in out['content']


def test_preview_short_singleline_shows_char_count_only():
    content = 'Short note.'
    out = preview_tool_args_for_terminal(
        'edit_note',
        {'note_title': 'T', 'content': content, 'mode': 'append'},
        content_preview_chars=80,
    )
    assert out['content'] == f'{len(content)} chars'


def test_preview_passes_through_other_tools():
    args = {'query': 'x'}
    assert preview_tool_args_for_terminal(
        'search_notes',
        args,
        content_preview_chars=APP_DEFAULTS.limits.tool_call_content_preview_chars,
    ) == args


def test_format_tool_call_line_wraps_long_args():
    long_query = 'x' * 200
    line = format_tool_call_line(
        'search_notes',
        {'query': long_query},
        width=60,
        max_arg_chars=512,
        content_preview_chars=80,
    )
    assert line.startswith('  ⚙  ')
    assert line.count('\n') >= 1
    assert 'search_notes' in line


def test_format_tool_call_line_create_note_no_body_preview():
    line = format_tool_call_line(
        'create_note',
        {
            'note_title': 'Big',
            'folder': 'Math',
            'content': 'word ' * 500,
        },
        width=80,
        max_arg_chars=512,
        content_preview_chars=80,
    )
    assert '2500 chars' in line
    assert 'word' not in line


def test_answer_stream_uses_repl_display_sink(ctx, monkeypatch):
    """REPL output pane renders panels without Rich Live cursor escapes."""
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)

    live_cls = MagicMock()
    monkeypatch.setattr('brain.ui.render.Live', live_cls)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}:md={markdown}\n',
    )

    begin_answer_stream()
    write_answer_delta('hello')
    end_answer_stream(final_content='hello')

    live_cls.assert_not_called()
    formatted = sink.get_transcript_formatted_text()
    assert 'PANEL:hello:md=False' in str(formatted)


def test_two_turn_ui_streams_stay_separate(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}:md={markdown}\n',
    )

    begin_answer_stream()
    write_answer_delta('Hello')
    end_answer_stream(final_content='Hello')
    sink.commit_turn()

    begin_answer_stream()
    write_answer_delta('World')
    end_answer_stream(final_content='World')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'PANEL:Hello:md=False' in transcript
    assert 'PANEL:World:md=False' in transcript
    assert 'PANEL:HelloWorld' not in transcript


def test_cancel_answer_stream_drops_empty_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    render = MagicMock(return_value='PANEL:empty\n')
    monkeypatch.setattr('brain.ui.render.render_answer_panel_to_ansi', render)

    begin_answer_stream()
    cancel_answer_stream()

    render.assert_not_called()
    assert str(sink.get_transcript_formatted_text()) == ''


def test_render_thinking_panel_to_ansi_smoke(ctx, monkeypatch):
    from rich.console import Console
    from rich.theme import Theme

    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr('brain.ui.render._panel_outer_width', lambda: 80)
    ctx.console = Console(
        theme=Theme(dict(ctx.defaults.ui.rich_theme)),
        force_terminal=True,
        color_system='standard',
    )
    from brain.ui.render import render_thinking_panel_to_ansi

    ansi = render_thinking_panel_to_ansi('Step one.\nStep two.')
    assert 'Step one.' in ansi
    assert '\x1b[' in ansi


def test_render_answer_panel_to_ansi_smoke(ctx, monkeypatch):
    from rich.console import Console
    from rich.theme import Theme

    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr('brain.ui.render._panel_outer_width', lambda: 80)
    ctx.console = Console(
        theme=Theme(dict(ctx.defaults.ui.rich_theme)),
        force_terminal=True,
        color_system='standard',
    )
    from brain.ui.render import render_answer_panel_to_ansi

    ansi = render_answer_panel_to_ansi('Hello', markdown=False)
    assert 'Hello' in ansi
    assert '\x1b[' in ansi


def test_end_thinking_stream_commits_filtered_display(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'THINK:{content}\n',
    )

    sink.begin_thinking_stream()
    sink.update_thinking_stream('Reasoning text', raw='Reasoning text')

    from brain.ui.render import end_thinking_stream

    end_thinking_stream(final_content='Reasoning text')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'THINK:Reasoning text' in transcript


def test_panel_line_counts_match_for_short_text(ctx, monkeypatch):
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr('brain.ui.render._panel_outer_width', lambda: 60)

    from brain.ui.render import render_answer_panel_to_ansi, render_thinking_panel_to_ansi

    answer = render_answer_panel_to_ansi('Hi.', markdown=False)
    thinking = render_thinking_panel_to_ansi('Hi.', markdown=False)
    answer_lines = len(answer.splitlines())
    thinking_lines = len(thinking.splitlines())
    assert answer_lines <= 8
    assert thinking_lines <= 8
    assert abs(answer_lines - thinking_lines) <= 1


def test_answer_finalize_keeps_same_panel_height_as_stream(ctx, monkeypatch):
    """Finalize must not switch to markdown and inflate the Second Brain panel."""
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr('brain.ui.render._panel_outer_width', lambda: 60)

    from brain.ui.render import render_answer_panel_to_ansi

    content = 'Short answer for height check.'
    streaming = render_answer_panel_to_ansi(content, markdown=False)
    finalized = render_answer_panel_to_ansi(content, markdown=False)
    assert len(finalized.splitlines()) == len(streaming.splitlines())
    markdown_finalize = render_answer_panel_to_ansi(content, markdown=True)
    assert len(markdown_finalize.splitlines()) >= len(streaming.splitlines())


def test_thinking_stream_commits_before_answer(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'THINK:{content}\n',
    )
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}:md={markdown}\n',
    )

    from brain.ui.render import (
        begin_answer_stream,
        begin_thinking_stream,
        end_answer_stream,
        end_thinking_stream,
        write_answer_delta,
        write_thinking_delta,
    )

    begin_thinking_stream()
    write_thinking_delta('Reasoning')
    end_thinking_stream()
    begin_answer_stream()
    write_answer_delta('Answer')
    end_answer_stream(final_content='Answer')
    sink.commit_turn()

    transcript = str(sink.get_transcript_formatted_text())
    assert 'THINK:Reasoning' in transcript
    assert 'PANEL:Answer:md=False' in transcript
    assert transcript.index('THINK:Reasoning') < transcript.index('PANEL:Answer')


def test_sync_thinking_display_stream_filters_scaffold(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'THINK:{content}\n',
    )

    from brain.ui.render import sync_thinking_display_stream

    raw = (
        "Here's a thinking process:\n\n"
        '1.  **Analyze User Input:** math\n\n'
        'Visible reasoning grows here.'
    )
    shown = ''
    shown = sync_thinking_display_stream(raw[:40], displayed_prefix=shown)
    assert shown == ''
    shown = sync_thinking_display_stream(raw, displayed_prefix=shown)
    assert 'Analyze User Input' not in shown
    assert 'Visible reasoning grows here.' in shown


def test_print_thinking_keeps_live_answer_until_finalize(ctx, monkeypatch):
    """``--think`` must not drop the streaming answer panel when reasoning renders."""
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}:md={markdown}\n',
    )
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'THINK:{content}\n',
    )

    begin_answer_stream()
    write_answer_delta('Hello there')
    print_thinking('Let me think…')

    assert sink.has_live_answer()
    transcript = str(sink.get_transcript_formatted_text())
    assert 'THINK:Let me think…' in transcript
    assert 'PANEL:Hello there' in transcript

    end_answer_stream(final_content='Hello there')
    sink.commit_turn()

    transcript = str(sink.get_transcript_formatted_text())
    assert transcript.count('PANEL:Hello there:md=False') == 1
    assert 'THINK:Let me think…' in transcript


def test_print_terminal_info_writes_to_stdout(ctx, monkeypatch):
    from io import StringIO

    from rich.console import Console
    from rich.theme import Theme

    from brain.ui.render import print_terminal_info

    tty = StringIO()
    monkeypatch.setattr('brain.ui.render.sys.__stdout__', tty)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    ctx.console = Console(theme=Theme(dict(ctx.defaults.ui.rich_theme)))
    print_terminal_info('Session saved. Goodbye.')
    assert 'Session saved. Goodbye.' in tty.getvalue()


def test_print_info_uses_dim_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_info_panel_to_ansi',
        lambda content: f'INFO-PANEL:{content}\n',
    )

    from brain.ui.render import print_info

    print_info('model: test')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'INFO-PANEL:model: test' in transcript


def test_end_answer_stream_commits_final_when_stream_inactive(ctx, monkeypatch):
    """Deferred finalize must not drop the panel when UI state was cleared early."""
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}:md={markdown}\n',
    )

    begin_answer_stream()
    write_answer_delta('Hello')
    cancel_answer_stream()

    end_answer_stream(final_content='Hello')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'PANEL:Hello:md=False' in transcript


def test_spinner_renders_in_transcript(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_spinner_to_ansi',
        lambda message, frame: f'SPINNER:{message}:{frame}\n',
    )

    sink.start_spinner('Thinking…', 'dots')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'SPINNER:Thinking…:0' in transcript


def test_spinner_glyph_uses_accent_style(ctx, monkeypatch):
    """REPL spinner glyph uses agent.spinner; message stays agent.info."""
    captured: list[str] = []
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: captured.append(markup) or 'ANSI\n',
    )

    from brain.ui.render import render_spinner_to_ansi

    render_spinner_to_ansi('Thinking…', 0)
    assert captured
    markup = captured[0]
    assert '[agent.spinner]' in markup
    assert '[agent.info]Thinking…[/]' in markup


def test_end_answer_stream_skips_blank_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    _wire_repl_sink(ctx, sink)
    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    render = MagicMock(return_value='PANEL: \n')
    monkeypatch.setattr('brain.ui.render.render_answer_panel_to_ansi', render)

    begin_answer_stream()
    end_answer_stream()

    render.assert_not_called()
    assert str(sink.get_transcript_formatted_text()) == ''


def test_repl_streaming_does_not_append_raw_tokens(ctx, monkeypatch):
    """Answer deltas must update the live panel, not append_raw per token."""
    from io import StringIO

    from rich.console import Console

    from brain.repl.display_port import RichConsoleDisplayPort

    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    ctx.repl_display = sink
    ctx.display = RichConsoleDisplayPort(
        Console(file=StringIO(), width=120),
    )
    append_raw = MagicMock(wraps=sink.append_raw)
    sink.append_raw = append_raw
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    begin_answer_stream()
    write_answer_delta('Hello')
    write_answer_delta(' world')
    end_answer_stream(final_content='Hello world')

    append_raw.assert_not_called()
    transcript = str(sink.get_transcript_formatted_text())
    assert 'PANEL:Hello world' in transcript

