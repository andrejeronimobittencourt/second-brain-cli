"""Tests for brain.core.session."""

from __future__ import annotations

from pathlib import Path

from brain.core.context_messages import truncate_messages_pair_aware
from brain.core.session import load_history, resolve_history_path, sanitize_session_name, save_history


class TestSessionPaths:
    def test_default_history(self, vault: Path):
        path = resolve_history_path(vault, '.agent_history.json', '')
        assert path == vault / '.agent_history.json'

    def test_named_session(self, vault: Path):
        path = resolve_history_path(vault, '.agent_history.json', 'research')
        assert path == vault / '.agent_sessions' / 'research.json'

    def test_sanitize_session_name(self):
        assert sanitize_session_name('my session!') == 'my_session_'


class TestSessionPersistence:
    def test_save_and_load(self, vault: Path):
        path = vault / 'test_session.json'
        msgs = [
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi'},
        ]
        save_history(path, msgs, max_messages=100, truncate_fn=truncate_messages_pair_aware)
        loaded = load_history(path, max_messages=100, truncate_fn=truncate_messages_pair_aware)
        assert loaded == msgs

    def test_atomic_save_valid_json(self, vault: Path):
        path = vault / 'atomic_session.json'
        msgs = [{'role': 'user', 'content': 'test'}]
        save_history(path, msgs, max_messages=100, truncate_fn=truncate_messages_pair_aware)
        assert path.is_file()
        assert not path.with_suffix(path.suffix + '.tmp').exists()
        raw = path.read_text(encoding='utf-8')
        assert raw.strip().startswith('[')
