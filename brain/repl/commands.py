"""REPL slash commands, session display seeding, and context stats formatting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from brain.ui import render as ui
from brain.agent.compression import trim_context
from brain.core.context import get_context
from brain.core.events import EventHandler
from brain.repl.context_meter import non_system_message_count, resolve_context_meter
from brain.repl.display import ReplDisplaySink
from brain.repl.input import ReplInput, refresh_repl_toolbar
from brain.vault.tools import active_vision_model, search_notes

SEARCH_PREFIX = '/search '
MODEL_PREFIX = '/model '
VISION_MODEL_PREFIX = '/vision-model '

SLASH_EXIT_COMMANDS = frozenset({'/exit', '/quit', '/bye'})

HELP_TEXT = """
Available slash commands:
  /help          — show this message
  /clear         — wipe conversation history and start fresh
  /history       — show the number of messages currently in context
  /stats         — context fill, model, and session file path
  /compact       — force-compress older turns into a summary now
  /search <term> — quick vault-wide search (bypasses the LLM)
  /model <name>  — switch Ollama chat model for this session
  /vision-model <name> — switch vision model for read_image this session
  /exit          — save session and exit (aliases: /quit, /bye)

Input (interactive REPL):
  Enter          — send message
  Esc Enter      — new line (multiline input)
  Ctrl+J         — new line (multiline input)
  Up / Down      — browse input history
  Ctrl+A / E     — start / end of line; Alt+B / F — word back / forward
  Ctrl+C         — save session and exit
  Ctrl+D         — save session and exit
  Page Up/Down   — scroll transcript
  Shift+Up/Down  — scroll transcript one line
  Ctrl+Up/Down   — scroll transcript one line
  Home / End     — top / bottom of transcript

With --think: Reasoning panel before the final answer (not tool-planning steps).

CLI: --config <file>  (or env SECOND_BRAIN_USER_CONFIG), --think, --resume,
     --vision-model, --host, --model, --session <name>, --print "prompt".

User JSON: ``system_prompt`` replaces the built-in system text when set.
           ``vault_instructions`` is always appended after that base for
           vault- or app-specific hints (e.g. editor plugins, link conventions).
"""


def format_history(messages: list[dict[str, Any]]) -> str:
    """Summarize message counts in the rolling context."""
    counts = {'user': 0, 'assistant': 0, 'tool': 0}
    for msg in messages:
        role = msg.get('role')
        if role in counts:
            counts[role] += 1
    total = sum(counts.values())
    cap = get_context().defaults.limits.max_context_messages
    return (
        f'{total} messages in rolling context (cap {cap}): '
        f'{counts["user"]} user, {counts["assistant"]} assistant, '
        f'{counts["tool"]} tool.'
    )


def format_stats(messages: list[dict[str, Any]]) -> str:
    """Summarize session stats."""
    ctx = get_context()
    count = non_system_message_count(messages)
    lim = ctx.defaults.limits.max_context_messages
    used, total = resolve_context_meter(
        last_prompt_tokens=ctx.last_prompt_tokens,
        num_ctx=ctx.num_ctx,
        message_count=count,
        message_cap=lim,
    )

    summary: list[str] = []
    if ctx.num_ctx > 0:
        fill_pct = (used / total * 100) if total else 0
        threshold_pct = ctx.defaults.context_compression.context_fill_ratio * 100
        summary = [
            f'{count} messages in context (cap: {lim})',
            (
                f'last prompt {used:,} / {total:,} tokens '
                f'({fill_pct:.0f}% — compresses at {threshold_pct:.0f}%)'
            ),
        ]
    elif total > 0:
        fill_pct = (used / total * 100) if total else 0
        summary = [f'context fill {fill_pct:.0f}% of message cap ({used}/{total})']
    else:
        summary = [f'{count} messages in context (cap: {lim})']

    parts = summary + [
        f'model: {ctx.user.ollama_model}',
        f'vision: {active_vision_model()}',
        f'vault: {ctx.vault_path.resolve()}',
        f'ollama: {ctx.user.ollama_host}',
        f'session: {ctx.history_path}',
    ]
    return '\n'.join(parts) + '.'


def final_assistant_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get('role') != 'assistant':
            continue
        if msg.get('tool_calls'):
            continue
        return (msg.get('content') or '').strip()
    return ''


def build_slash_handlers(
    *,
    system_msg: dict[str, Any],
    messages: list[dict[str, Any]],
    on_event: EventHandler | None,
) -> dict[str, Callable[[], None]]:
    """Return slash-command handlers that mutate the shared ``messages`` list."""

    def _slash_help() -> None:
        ui.print_info(HELP_TEXT.strip())

    def _slash_clear() -> None:
        ctx = get_context()
        messages.clear()
        messages.append(system_msg)
        if ctx.history_path.exists():
            ctx.history_path.unlink()
        sink = ctx.repl_display
        if sink is not None:
            sink.clear_session()
        ui.print_info('Conversation cleared.')
        refresh_repl_toolbar()

    def _slash_history() -> None:
        ui.print_info(format_history(messages))

    def _slash_stats() -> None:
        ui.print_info(format_stats(messages))

    def _slash_compact() -> None:
        before = sum(1 for m in messages if m.get('role') != 'system')
        trimmed = trim_context(messages, force=True, on_event=on_event)
        messages.clear()
        messages.extend(trimmed)
        after = sum(1 for m in messages if m.get('role') != 'system')
        ui.print_info(f'Compacted context: {before} → {after} messages.')
        refresh_repl_toolbar()

    return {
        '/help': _slash_help,
        '/clear': _slash_clear,
        '/history': _slash_history,
        '/stats': _slash_stats,
        '/compact': _slash_compact,
    }


def seed_repl_display(
    sink: ReplDisplaySink,
    *,
    repl: ReplInput | None,
    resume: bool,
    messages: list[dict[str, Any]],
) -> None:
    """Append startup banners to the REPL transcript."""
    ctx = get_context()
    sink.append_session_rich(
        "[agent.info]Second Brain CLI — Type '/help' for commands and input keys.[/]\n",
    )
    if ctx.vault_path_defaulted:
        sink.append_session_rich(
            '[agent.error]✗  vault_path is not set in your user JSON — defaulting to the '
            'script directory. Set vault_path to your Markdown vault folder.[/]',
        )
    if repl is not None and not repl.uses_line_editor:
        reason = repl.fallback_reason or 'unknown'
        sink.append_session_rich(
            f'[agent.info]Line editor unavailable ({reason}) — using basic input. '
            'Install prompt_toolkit for arrow keys and history.[/]\n',
        )
    if resume:
        prior_count = sum(1 for m in messages if m.get('role') != 'system')
        if prior_count:
            sink.append_session_rich(
                f'[agent.info]Resumed previous session ({prior_count} messages loaded).[/]\n',
            )
        else:
            sink.append_session_rich(
                '[agent.info]No saved session found — starting fresh.[/]\n',
            )


def handle_prefixed_command(user_input: str) -> bool:
    """
    Handle ``/search``, ``/model``, and ``/vision-model`` prefixes.

    Returns True when the line was consumed as a command.
    """
    ctx = get_context()

    if user_input.startswith(SEARCH_PREFIX):
        query = user_input[len(SEARCH_PREFIX):].strip()
        if query:
            ui.print_info(str(search_notes(query)))
        else:
            ui.print_error(f'Usage: {SEARCH_PREFIX}<term>')
        return True

    if user_input.startswith(MODEL_PREFIX):
        model_name = user_input[len(MODEL_PREFIX):].strip()
        if model_name:
            ctx.user = replace(ctx.user, ollama_model=model_name)
            ui.print_info(f'Chat model switched to {model_name}.')
        else:
            ui.print_error(f'Usage: {MODEL_PREFIX}<model-name>')
        return True

    if user_input.startswith(VISION_MODEL_PREFIX):
        vision_name = user_input[len(VISION_MODEL_PREFIX):].strip()
        if vision_name:
            ctx.vision_model_override = vision_name
            ui.print_info(f'Vision model switched to {vision_name}.')
        else:
            ui.print_error(f'Usage: {VISION_MODEL_PREFIX}<model-name>')
        return True

    return False
