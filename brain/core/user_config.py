"""
User / environment-specific settings loaded from JSON only.

Keep this schema small: paths, Ollama connection, and optional prompt overrides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass
class UserConfig:
    """Fields map 1:1 to ``second_brain_user.json`` keys."""

    vault_path: str = ''
    ollama_host: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'gemma4:latest'
    ollama_vision_model: str = ''
    history_filename: str = '.agent_history.json'
    note_encoding: str = 'utf-8'
    system_prompt: str = ''
    vault_instructions: str = ''
    log_level: str = ''
    log_file: str = ''

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserConfig:
        """Build from a JSON object, ignoring unknown keys and ``_`` comments."""
        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, val in data.items():
            if key.startswith('_') or key not in allowed:
                continue
            kwargs[key] = val
        return cls(**kwargs)

    def merged_with_cli(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
    ) -> 'UserConfig':
        """Return a copy with non-empty CLI overrides applied (one-shot)."""
        overrides: dict[str, Any] = {}
        if host and host.strip():
            overrides['ollama_host'] = host.strip()
        if model and model.strip():
            overrides['ollama_model'] = model.strip()
        return replace(self, **overrides)


def load_user_config_file(path: Path) -> UserConfig:
    """Parse JSON file into ``UserConfig``."""
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('User config root must be a JSON object')
    return UserConfig.from_dict(raw)
