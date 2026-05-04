"""
Runtime application context — wires user config, immutable defaults, and clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ollama import Client

from brain.defaults import AppDefaults
from brain.user_config import UserConfig

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class ApplicationContext:
    """Live state for one process; built in ``brain.bootstrap``."""

    user: UserConfig
    defaults: AppDefaults
    vault_path: Path
    history_path: Path
    ollama_client: Client
    latex_pairs: tuple[tuple[str, str], ...]
    channel_leak: re.Pattern[str]
    console: Console | None = None
    config_source: str = ''
    vision_model_override: str = ''
    vault_path_defaulted: bool = False


_CTX: Optional[ApplicationContext] = None


def get_context() -> ApplicationContext:
    """Return the active context (must call ``bootstrap`` first)."""
    if _CTX is None:
        raise RuntimeError('ApplicationContext not initialised; call bootstrap() first')
    return _CTX


def set_context(ctx: ApplicationContext) -> None:
    """Install context (used by ``bootstrap``)."""
    global _CTX
    _CTX = ctx
