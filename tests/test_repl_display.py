"""Tests for brain.repl.display."""

from __future__ import annotations

from unittest.mock import MagicMock

from brain.repl.display import ReplDisplaySink
from brain.repl.display_port import ReplDisplayPort


def test_stream_updates_render_panel_without_live(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    ctx.repl_display = sink
    ctx.display = ReplDisplayPort(sink=sink)

    render = MagicMock(return_value='[panel]hello[/panel]\n')
    monkeypatch.setattr('brain.ui.render.render_answer_panel_to_ansi', render)

    sink.begin_answer_stream()
    sink.update_answer_stream('hello', markdown=False)
    sink.end_answer_stream('hello')

    render.assert_any_call('hello', markdown=False)
    assert render.call_count == 2
    assert all(call.kwargs.get('markdown') is False for call in render.call_args_list)
    text = sink.get_transcript_formatted_text()
    assert '[panel]hello[/panel]' in str(text)
    invalidate.assert_called()


def test_commit_turn_accumulates_session_history(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.begin_answer_stream()
    sink.end_answer_stream('first')
    sink.commit_turn()

    sink.begin_answer_stream()
    sink.end_answer_stream('second')
    sink.commit_turn()

    transcript = str(sink.get_transcript_formatted_text())
    assert 'PANEL:first' in transcript
    assert 'PANEL:second' in transcript
    assert 'PANEL:firstsecond' not in transcript
    assert 'firstsecond' not in transcript


def test_commit_turn_preserves_uncommitted_live_stream(ctx, monkeypatch):
    """Uncommitted live preview is kept when the turn ends without ``end_answer_stream``."""
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.begin_answer_stream()
    sink.update_answer_stream('live-only', markdown=False)
    sink.commit_turn()

    transcript = str(sink.get_transcript_formatted_text())
    assert transcript.count('PANEL:live-only') == 1


def test_commit_turn_does_not_duplicate_committed_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.begin_answer_stream()
    sink.update_answer_stream('done', markdown=False)
    sink.end_answer_stream('done')
    sink.commit_turn()

    transcript = str(sink.get_transcript_formatted_text())
    assert transcript.count('PANEL:done') == 1


def test_user_prompt_appears_in_transcript(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: f'{markup}\n',
    )
    monkeypatch.setattr('brain.ui.render.repl_transcript_inner_width', lambda: 40)
    monkeypatch.setattr(
        'brain.core.sanitize.prewrap_for_terminal',
        lambda text, _inner, _d: text,
    )

    sink.append_user_prompt('find my notes on rust')

    transcript = str(sink.get_transcript_formatted_text())
    assert 'You:' in transcript
    assert 'find my notes on rust' in transcript


def test_long_user_prompt_is_wrapped(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr('brain.ui.render.repl_transcript_inner_width', lambda: 20)
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: f'{markup}\n',
    )

    long_text = 'word ' * 30
    sink.append_user_prompt(long_text.strip())

    transcript = str(sink.get_transcript_formatted_text())
    assert '    word' in transcript or transcript.count('word') > 3


def test_transcript_line_count(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: f'{markup}\n',
    )

    assert sink.transcript_line_count() == 0

    sink.append_session_raw('line one\nline two\n')
    assert sink.transcript_line_count() == 2


def test_visible_transcript_scroll_slice(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    lines = [f'line {index}' for index in range(20)]
    sink.append_session_raw('\n'.join(lines) + '\n')

    text, total = sink.visible_transcript_ansi(scroll=0, viewport=5)
    assert total == 20
    assert 'line 0' in text
    assert '↓' in text
    assert 'line 4' not in text

    text, _total = sink.visible_transcript_ansi(scroll=_max_scroll(sink, 5), viewport=5)
    assert '↑' in text
    assert 'line 19' in text


def _max_scroll(sink: ReplDisplaySink, viewport: int) -> int:
    total = sink.transcript_line_count()
    if total <= viewport:
        return 0
    return max(0, total - max(1, viewport - 1))


def test_cancel_answer_stream_drops_live_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_answer_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.begin_answer_stream()
    sink.update_answer_stream('partial', markdown=False)
    sink.cancel_answer_stream()

    assert str(sink.get_transcript_formatted_text()) == ''


def test_rich_capture_console_applies_theme(ctx, monkeypatch):
    """REPL ANSI capture must inherit agent.info / agent.tool styles."""
    from rich.console import Console
    from rich.theme import Theme

    from brain.ui.render import render_markup_to_ansi

    monkeypatch.setattr('brain.ui.render.RICH_AVAILABLE', True)
    monkeypatch.setattr('brain.ui.render._panel_outer_width', lambda: 80)
    ctx.console = Console(
        theme=Theme(dict(ctx.defaults.ui.rich_theme)),
        force_terminal=True,
        color_system='standard',
    )
    ansi = render_markup_to_ansi('[agent.info]dim status[/]')
    assert 'dim status' in ansi
    assert '\x1b[' in ansi


def test_append_info_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_info_panel_to_ansi',
        lambda content: f'PANEL:{content}\n',
    )

    sink.append_info_panel('stats line')

    assert 'PANEL:stats line' in str(sink.get_transcript_formatted_text())


def test_tool_line_not_duplicated_in_activity_strip(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: f'ANSI:{markup}\n',
    )

    sink.append_rich('  [agent.tool]⚙  search_notes(query="x")[/]')

    assert 'search_notes' in str(sink.get_transcript_formatted_text())


def test_visible_transcript_plain_strips_ansi(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_markup_to_ansi',
        lambda markup: f'\x1b[1m{markup}\x1b[0m\n',
    )

    sink.append_session_raw('visible line\n')

    plain, total = sink.visible_transcript_plain(scroll=0, viewport=10)
    assert total == 1
    assert plain == 'visible line'
    assert '\x1b[' not in plain


def test_revert_thinking_commit_removes_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.end_thinking_stream(final_content='planning reasoning')
    assert 'PANEL:planning reasoning' in str(sink.get_transcript_formatted_text())

    sink.revert_thinking_commit()

    assert str(sink.get_transcript_formatted_text()) == ''


def test_end_thinking_stream_replaces_existing_panel(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr(
        'brain.ui.render.render_thinking_panel_to_ansi',
        lambda content, markdown=False: f'PANEL:{content}\n',
    )

    sink.end_thinking_stream(final_content='first')
    sink.end_thinking_stream(final_content='second')

    transcript = str(sink.get_transcript_formatted_text())
    assert transcript.count('PANEL:first') == 0
    assert transcript.count('PANEL:second') == 1


def test_spinner_appears_in_compose_without_rich(ctx, monkeypatch):
    invalidate = MagicMock()
    sink = ReplDisplaySink(invalidate=invalidate)
    monkeypatch.setattr('brain.ui.render.render_spinner_to_ansi', lambda *_a, **_k: '')

    sink.start_spinner('Working…', 'dots')
    plain, _total = sink.visible_transcript_plain(scroll=0, viewport=10)

    assert 'Working' in plain
    assert plain.strip()


def test_ansi_cursor_strip_preserves_sgr_colors_and_text():
    from brain.ui.render import _plain_from_ansi, _strip_ansi_cursor_motion

    colored = '\x1b[1;31mHello\x1b[0m world\n'
    assert 'Hello' in _strip_ansi_cursor_motion(colored)
    assert 'Hello world' in _plain_from_ansi(colored)
    assert '\x1b[31m' not in _plain_from_ansi(colored)
