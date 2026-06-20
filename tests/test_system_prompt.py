"""Tests for brain.core.system_prompt and system_context."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from brain.core.system_context import system_context_footer
from brain.core.system_prompt import build_system_prompt
from brain.core.user_config import UserConfig


class TestSystemContextFooter:
    def test_includes_time_and_vault(self, vault: Path):
        fixed = datetime(2026, 6, 14, 15, 30, tzinfo=timezone.utc)
        footer = system_context_footer(vault_path=vault, now=fixed)
        assert 'Current local time:' in footer
        assert 'Vault root:' in footer
        assert vault.as_posix() in footer.replace('\\', '/')


class TestBuildSystemPrompt:
    def test_footer_always_last(self, vault: Path):
        user = UserConfig(
            system_prompt='Custom base.',
            vault_instructions='Extra vault rules.',
        )
        prompt = build_system_prompt(user=user, vault_path=vault)
        assert prompt.startswith('Custom base.')
        assert 'Extra vault rules.' in prompt
        assert prompt.rstrip().endswith(vault.resolve().as_posix().replace('\\', '/'))

    def test_default_includes_footer(self, vault: Path):
        user = UserConfig()
        prompt = build_system_prompt(user=user, vault_path=vault)
        assert 'Current local time:' in prompt
        assert 'autonomous assistant' in prompt
