#!/usr/bin/env python3
"""
Second Brain CLI — local Ollama + Markdown vault paths.

User-only JSON (paths, Ollama, optional system prompt): copy
``second_brain_user.example.json`` → ``second_brain_user.json`` beside this script,
or pass ``--config /path/to/user.json``, or set ``SECOND_BRAIN_USER_CONFIG``.

Run:  python agent.py [--config FILE] [--resume] [--think] [--vision-model TAG]
      [--host URL] [--model NAME] [--session NAME] [--print "prompt"]

Requires: pip install ollama rich prompt_toolkit PyYAML
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brain import bootstrap, run_agent, ui

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Second Brain CLI — local Ollama + Markdown vault assistant.',
    )
    parser.add_argument(
        '--config',
        default='',
        metavar='FILE',
        help=(
            'Path to second_brain_user.json (defaults: env SECOND_BRAIN_USER_CONFIG, '
            'then ./second_brain_user.json beside agent.py).'
        ),
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Continue from the saved session file (history_filename in user JSON).',
    )
    parser.add_argument(
        '--think',
        action='store_true',
        help='Show model reasoning before each answer (Ollama think=true).',
    )
    parser.add_argument(
        '--vision-model',
        default='',
        metavar='MODEL',
        help=(
            'Override vision model for read_image for this run (else user JSON / '
            'OLLAMA_VISION_MODEL / main model).'
        ),
    )
    parser.add_argument(
        '--host',
        default='',
        metavar='URL',
        help='Override Ollama API base URL for this run (else user JSON ollama_host).',
    )
    parser.add_argument(
        '--model',
        default='',
        metavar='NAME',
        help=(
            'Override main chat model for this run (else user JSON ollama_model; '
            'e.g. gemma4:latest).'
        ),
    )
    parser.add_argument(
        '--session',
        default='',
        metavar='NAME',
        help=(
            'Use a named session file under {vault}/.agent_sessions/ instead of '
            'the default history file.'
        ),
    )
    parser.add_argument(
        '--print',
        default='',
        metavar='PROMPT',
        dest='print_prompt',
        help='Run one prompt and print the answer to stdout (no interactive REPL).',
    )
    cli = parser.parse_args()

    cfg_arg = Path(cli.config).expanduser().resolve() if cli.config.strip() else None
    if cfg_arg is not None and not cfg_arg.is_file():
        print(f'Config file not found: {cfg_arg}', file=sys.stderr)
        sys.exit(1)

    ctx = bootstrap(
        cfg_arg,
        SCRIPT_DIR,
        cli_host=cli.host.strip(),
        cli_model=cli.model.strip(),
        session_name=cli.session.strip(),
    )

    if not ctx.vault_path.exists():
        ui.print_error(
            f"Vault path '{ctx.vault_path}' does not exist. Set vault_path in your user "
            'JSON to your Markdown vault root (the folder that contains your `.md` notes).',
        )
        sys.exit(1)

    exit_code = run_agent(
        resume=cli.resume,
        think=cli.think,
        vision_model=cli.vision_model,
        print_prompt=cli.print_prompt.strip(),
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
