"""
Compose ``UserConfig`` + ``AppDefaults`` into ``ApplicationContext`` and Rich console.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

from ollama import Client

from brain.context import ApplicationContext, set_context
from brain.defaults import APP_DEFAULTS
from brain.logging_setup import configure_logging
from brain.user_config import UserConfig, load_user_config_file

try:
    from rich.console import Console
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _discover_user_config(
    explicit: Optional[Path],
    script_dir: Path,
) -> tuple[Optional[Path], UserConfig]:
    """Pick first existing user JSON and load it, else defaults."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_p = os.environ.get('SECOND_BRAIN_USER_CONFIG', '').strip()
    if env_p:
        candidates.append(Path(env_p).expanduser())
    candidates.append(script_dir / 'second_brain_user.json')

    for path in candidates:
        if path.is_file():
            return path, load_user_config_file(path)
    return None, UserConfig()


def _init_console(ctx: ApplicationContext) -> None:
    if not RICH_AVAILABLE:
        return
    theme = dict(ctx.defaults.ui.rich_theme)
    ctx.console = Console(
        theme=Theme(theme),
        highlight=True,
        soft_wrap=ctx.defaults.terminal.rich_soft_wrap,
    )


def bootstrap(
    config_file: Optional[Path],
    script_dir: Path,
    *,
    cli_host: str = '',
    cli_model: str = '',
) -> ApplicationContext:
    """
    Load user JSON (optional), apply CLI overrides, build ``ApplicationContext``.

    ``config_file`` — if set, must exist (caller validates).
    """
    loaded_path, user = _discover_user_config(config_file, script_dir)
    user = user.merged_with_cli(
        host=cli_host or None,
        model=cli_model or None,
    )
    configure_logging(user.log_level, user.log_file)

    d = APP_DEFAULTS
    vault_s = user.vault_path.strip()
    vault_path_defaulted = not vault_s
    vault_path = Path(vault_s).expanduser().resolve() if vault_s else script_dir.resolve()
    history_path = vault_path / (user.history_filename or '.agent_history.json')

    client = Client(host=user.ollama_host)

    try:
        model_list = client.list()
        local_names = {
            m.model for m in (model_list.models or [])
        } if hasattr(model_list, 'models') else set()
        if local_names and user.ollama_model not in local_names:
            short_names = {n.split(':')[0] for n in local_names}
            if user.ollama_model.split(':')[0] not in short_names:
                print(
                    f"Warning: Model '{user.ollama_model}' not found locally. "
                    f'Available: {", ".join(sorted(local_names))}. '
                    'Ollama may pull it on first use.',
                    file=sys.stderr,
                )
    except Exception as exc:
        print(
            f"Cannot reach Ollama at {user.ollama_host}: {exc}\n"
            "Is 'ollama serve' running?",
            file=sys.stderr,
        )
        sys.exit(1)

    leak = re.compile(d.model.channel_leak_regex, re.IGNORECASE)
    latex_pairs = d.latex_symbol_pairs

    source = str(loaded_path) if loaded_path else '(no user JSON; factory defaults)'

    ctx = ApplicationContext(
        user=user,
        defaults=d,
        vault_path=vault_path,
        history_path=history_path,
        ollama_client=client,
        latex_pairs=latex_pairs,
        channel_leak=leak,
        config_source=source,
        vault_path_defaulted=vault_path_defaulted,
    )
    _init_console(ctx)
    set_context(ctx)
    return ctx
