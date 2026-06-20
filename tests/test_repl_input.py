"""Tests for brain.repl.input."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.repl.input import (
    ReplInput,
    ReplStatus,
    format_toolbar,
    input_history_path,
    repl_input_is_locked,
    repl_prompt_prefix_text,
    set_repl_awaiting_final_answer,
)


class TestFormatToolbar:
    def test_includes_vault_model_session(self, vault: Path):
        status = ReplStatus(
            vault_path=vault,
            chat_model='gemma4:latest',
            vision_model='llava:latest',
            session_path=vault / '.agent_sessions' / 'research.json',
        )
        text = format_toolbar(status)
        assert 'vault:' in text
        assert 'gemma4:latest' in text
        assert 'llava:latest' in text
        assert 'research.json' in text

    def test_omits_context_progress(self, vault: Path):
        status = ReplStatus(
            vault_path=vault,
            chat_model='gemma4:latest',
            vision_model='llava:latest',
            session_path=vault / '.agent_sessions' / 'research.json',
        )
        text = format_toolbar(status)
        assert '0%' not in text
        assert '25%' not in text
        assert text.startswith(' vault:')


class TestInputHistoryPath:
    def test_under_vault(self, ctx, vault: Path):
        assert input_history_path(vault) == vault / '.agent_input_history'


class TestReplInputFallback:
    def test_skips_prompt_toolkit_when_not_tty(self, ctx, vault: Path, monkeypatch):
        monkeypatch.setattr('brain.repl.input.PROMPT_TOOLKIT_AVAILABLE', True)
        monkeypatch.setattr('sys.stdin', type('Stdin', (), {'isatty': lambda self: False})())
        repl = ReplInput(vault)
        assert not repl.uses_line_editor

    def test_uses_input_when_not_tty(self, ctx, vault: Path, monkeypatch):
        monkeypatch.setattr('brain.repl.input.PROMPT_TOOLKIT_AVAILABLE', True)
        monkeypatch.setattr('sys.stdin', type('Stdin', (), {'isatty': lambda self: False})())
        monkeypatch.setattr('builtins.input', lambda _prompt: '  hello  ')
        repl = ReplInput(vault)
        status = ReplStatus(
            vault_path=vault,
            chat_model='m',
            vision_model='v',
            session_path=vault / '.agent_history.json',
        )
        assert repl.prompt(status=status) == 'hello'

    def test_eof_returns_none(self, ctx, vault: Path, monkeypatch):
        monkeypatch.setattr('brain.repl.input.PROMPT_TOOLKIT_AVAILABLE', False)
        monkeypatch.setattr('sys.stdin', type('Stdin', (), {'isatty': lambda self: False})())

        def _eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr('builtins.input', _eof)
        repl = ReplInput(vault)
        status = ReplStatus(
            vault_path=vault,
            chat_model='m',
            vision_model='v',
            session_path=vault / '.agent_history.json',
        )
        assert repl.prompt(status=status) is None


class TestReplFarewell:
    def test_farewell_printed_after_repl_exit(self, ctx, vault: Path, monkeypatch):
        """Farewell line appears once the REPL application returns (e.g. after Ctrl+C)."""
        printed: list[str] = []

        def _fake_run_application(
            self: ReplInput,
            *,
            on_exit: object,
            **_: object,
        ) -> None:
            assert callable(on_exit)
            on_exit()

        monkeypatch.setattr(ReplInput, '_run_application', _fake_run_application)
        monkeypatch.setattr(
            'brain.ui.render.print_terminal_info',
            lambda msg: printed.append(msg),
        )

        repl = ReplInput(vault)
        repl._uses_ptk = True
        status = ReplStatus(
            vault_path=vault,
            chat_model='m',
            vision_model='v',
            session_path=vault / '.agent_history.json',
        )
        repl.run(
            status_fn=lambda: status,
            process_line=lambda _: None,
            on_exit=lambda: None,
            farewell_message='Session saved. Goodbye.',
        )
        assert printed == ['Session saved. Goodbye.']

    def test_safe_app_exit_ignores_duplicate(self):
        from brain.repl.input import _safe_app_exit

        class _FakeApp:
            def __init__(self) -> None:
                self.calls = 0

            def exit(self) -> None:
                self.calls += 1
                if self.calls > 1:
                    raise Exception('Return value already set. Application.exit() failed.')

        app = _FakeApp()
        _safe_app_exit(app)
        _safe_app_exit(app)
        assert app.calls == 2


class TestReplInputLock:
    def test_awaiting_final_answer_flag(self, ctx):
        assert not repl_input_is_locked()
        set_repl_awaiting_final_answer(True)
        assert repl_input_is_locked()
        assert ctx.repl_awaiting_final_answer
        set_repl_awaiting_final_answer(False)
        assert not repl_input_is_locked()


class TestPromptPrefix:
    def test_label_only_on_first_row(self):
        assert repl_prompt_prefix_text('You', 0, 0) == 'You: '
        assert repl_prompt_prefix_text('You', 0, 1) == '     '
        assert repl_prompt_prefix_text('You', 1, 0) == '     '


class TestReplKeyBindings:
    def test_builds_without_invalid_keys(self):
        from brain.repl import input as ri

        if not ri.PROMPT_TOOLKIT_AVAILABLE:
            return
        kb = ri._repl_key_bindings(multiline=True, on_ctrl_c=lambda: None)
        assert kb is not None
        kb_single = ri._repl_key_bindings(multiline=False, on_ctrl_c=lambda: None)
        assert kb_single is not None

    def test_builds_with_busy_filter(self):
        from brain.repl import input as ri

        if not ri.PROMPT_TOOLKIT_AVAILABLE:
            pytest.skip('prompt_toolkit not installed')
        kb = ri._repl_key_bindings(
            multiline=True,
            on_ctrl_c=lambda: None,
            is_busy=lambda: True,
        )
        assert kb is not None


class TestKeyboardInterrupt:
    def test_propagates_from_basic_input(self, ctx, vault: Path, monkeypatch):
        monkeypatch.setattr('brain.repl.input.PROMPT_TOOLKIT_AVAILABLE', False)
        monkeypatch.setattr('sys.stdin', type('Stdin', (), {'isatty': lambda self: False})())

        def _interrupt(_prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr('builtins.input', _interrupt)
        repl = ReplInput(vault)
        status = ReplStatus(
            vault_path=vault,
            chat_model='m',
            vision_model='v',
            session_path=vault / '.agent_history.json',
        )
        with pytest.raises(KeyboardInterrupt):
            repl.prompt(status=status)
