"""
Runtime application context — wires user config, immutable defaults, and clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from ollama import Client

from brain.core.defaults import AppDefaults
from brain.core.user_config import UserConfig

if TYPE_CHECKING:
    from rich.console import Console

    from brain.repl.display import ReplDisplaySink


@dataclass
class ApplicationContext:
    """Live state for one process; built in ``brain.core.bootstrap``."""

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
    # Context-window size fetched from Ollama at startup (0 = unknown / not fetched).
    num_ctx: int = 0
    # prompt_eval_count from the most recent Ollama response; used to gauge token fill.
    last_prompt_tokens: int = 0
    # True when the last final assistant reply was streamed live to the terminal.
    last_answer_streamed_to_tty: bool = False
    # True when reasoning was streamed live to the REPL transcript.
    last_thinking_streamed_to_tty: bool = False
    # True after the Reasoning panel was committed for the current tool-loop turn.
    thinking_committed_this_turn: bool = False
    # True when the interactive REPL uses prompt_toolkit (affects answer rendering).
    repl_line_editor: bool = False
    # Terminal width last measured by the REPL layout (safe from worker threads).
    repl_terminal_columns: int = 0
    # True from user submit until the full agent turn finishes (final answer).
    repl_awaiting_final_answer: bool = False
    # Optional hook to repaint the prompt_toolkit UI (set by ReplInput).
    repl_ui_refresh: Callable[[], None] | None = None
    # Active REPL output pane sink (set while the Application is running).
    repl_display: ReplDisplaySink | None = None
    # True during ``--print`` one-shot CLI (no interactive REPL).
    print_mode: bool = False
    # Single-owner display layer (REPL, print mode, or null before startup).
    display: Any = None


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
