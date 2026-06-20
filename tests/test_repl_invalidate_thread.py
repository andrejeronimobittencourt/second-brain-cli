"""Verify REPL repaint scheduling from a worker thread."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from brain.repl.display import ReplDisplaySink
from brain.repl import input as repl_input


def test_repaint_callback_runs_from_worker_thread():
    """Display sink updates must repaint even when get_app() is unavailable."""
    calls: list[str] = []

    def invalidate():
        calls.append('repaint')

    sink = ReplDisplaySink(invalidate=invalidate)
    sink.start_spinner('Working…', 'dots')

    def worker():
        sink.update_answer_stream('Hello', markdown=False)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2)
    assert calls, 'expected at least one repaint after worker update'


def test_schedule_ui_callback_uses_app_loop_when_running():
    """Buffer unlock callbacks must reach the UI thread while the app runs."""
    calls: list[str] = []
    app = MagicMock()
    app._is_running = True
    app.loop = MagicMock()

    repl_input._APP_HOLDER['app'] = app
    try:
        repl_input._schedule_ui_callback(lambda: calls.append('ok'))
    finally:
        repl_input._APP_HOLDER['app'] = None

    app.loop.call_soon_threadsafe.assert_called_once()
    scheduled = app.loop.call_soon_threadsafe.call_args[0][0]
    scheduled()
    assert calls == ['ok']


def test_invalidate_ui_schedules_on_ui_loop_from_worker():
    """Worker-thread repaints must use the Application event loop."""
    app = MagicMock()
    app._is_running = True
    app._loop_thread = object()  # not the current thread
    app.loop = MagicMock()

    repl_input._APP_HOLDER['app'] = app
    try:
        repl_input._invalidate_ui()
    finally:
        repl_input._APP_HOLDER['app'] = None

    app.loop.call_soon_threadsafe.assert_called_once_with(app.invalidate)


def test_invalidate_ui_calls_directly_on_ui_thread():
    """UI-thread repaints call invalidate immediately."""
    app = MagicMock()
    app._is_running = True
    app._loop_thread = threading.current_thread()

    repl_input._APP_HOLDER['app'] = app
    try:
        repl_input._invalidate_ui()
    finally:
        repl_input._APP_HOLDER['app'] = None

    app.invalidate.assert_called_once()
    app.loop.call_soon_threadsafe.assert_not_called()
