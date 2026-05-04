"""Shared fixtures: temp vault + ApplicationContext for tool tests."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.context import ApplicationContext, set_context
from brain.defaults import APP_DEFAULTS
from brain.tools import _NOTE_INDEX
from brain.user_config import UserConfig


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Create a minimal Markdown vault with a few notes."""
    (tmp_path / 'Note A.md').write_text('# Note A\nHello world\n', encoding='utf-8')
    (tmp_path / 'Note B.md').write_text(
        '---\ntags: [python, testing]\n---\n\nLinks to [[Note A]].\n',
        encoding='utf-8',
    )
    sub = tmp_path / 'Subfolder'
    sub.mkdir()
    (sub / 'Deep Note.md').write_text('# Deep\nNested content.\n', encoding='utf-8')
    return tmp_path


@pytest.fixture()
def ctx(vault: Path) -> ApplicationContext:
    """Install an ApplicationContext backed by the temp vault."""
    user = UserConfig(vault_path=str(vault))
    d = replace(
        APP_DEFAULTS,
        context_compression=replace(APP_DEFAULTS.context_compression, enabled=False),
    )
    leak = re.compile(d.model.channel_leak_regex, re.IGNORECASE)

    context = ApplicationContext(
        user=user,
        defaults=d,
        vault_path=vault,
        history_path=vault / '.agent_history.json',
        ollama_client=MagicMock(),
        latex_pairs=d.latex_symbol_pairs,
        channel_leak=leak,
    )
    set_context(context)
    _NOTE_INDEX.invalidate()
    return context


@pytest.fixture()
def ctx_compress(vault: Path) -> ApplicationContext:
    """
    Like ``ctx`` but with a small context cap and rolling summary enabled,
    for tests that exercise ``_trim_context`` compression.
    """
    user = UserConfig(vault_path=str(vault))
    lim = replace(APP_DEFAULTS.limits, max_context_messages=6)
    cc = replace(APP_DEFAULTS.context_compression, enabled=True)
    d = replace(APP_DEFAULTS, limits=lim, context_compression=cc)
    leak = re.compile(d.model.channel_leak_regex, re.IGNORECASE)
    client = MagicMock()
    client.chat.return_value = {
        'message': {
            'role': 'assistant',
            'content': 'Prior: user asked about vault notes and search.',
        },
    }
    context = ApplicationContext(
        user=user,
        defaults=d,
        vault_path=vault,
        history_path=vault / '.agent_history.json',
        ollama_client=client,
        latex_pairs=d.latex_symbol_pairs,
        channel_leak=leak,
    )
    set_context(context)
    _NOTE_INDEX.invalidate()
    return context
